import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import nltk
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow_core.python.keras.models import load_model

from utils.data_utils import DataUtility as du
from utils.model_utils import CNNUtils as cnn_u

nltk.download('stopwords')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# In[98]:

# Load data
tweets = du.load_data('../data/tweets_dataset.csv')
sentiments = du.load_data('../data/tweets_sentiments.csv')
# Remove headers

du.remove_data_header(tweets)
du.remove_data_header(sentiments)
du.remove_data_header(tweets)
# Get only cells with tweets text and labels
tweets_text_list, tweets_label_list = du.get_selected_columns(tweets, 1, 8)
# Filter data
filtered_tweets, filtered_words_list = du.filter_data(tweets_text_list)

# In[]:
# Eliminate the tweets that don't have a label
filtered_tweets, tweets_label_list, sentiments, unlabeled_tweets, unlabeled_sentiments = du.separate_unlabeled_data(
    filtered_tweets, tweets_label_list, sentiments)
# Convert labels from string to int
tweets_label_list = du.convert_labels_to_int(tweets_label_list)
# In[]:
print(sentiments)
for i in range(0, len(sentiments)):
    sentiments[i] = du.convert_labels_to_int(sentiments[i])

print(len(sentiments))
print(len(filtered_tweets))
# In[]:

# Split data into train and test
df = pd.DataFrame({"tweet": filtered_tweets, "label": tweets_label_list, "sentiment": sentiments})
target = df['label'].values.tolist()
data_text = df['tweet'].values.tolist()
data_sentiments = df['sentiment'].values.tolist()
x_train, x_test, y_train, y_test, sentiments_train, sentiments_test = train_test_split(data_text, target,
                                                                                       data_sentiments, test_size=0.2)
# In[]:
print(df)

# In[]:
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data_text)

# Training set token with text_to_sequences.

x_train_tokens = tokenizer.texts_to_sequences(x_train)
x_test_tokens = tokenizer.texts_to_sequences(x_test)

num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)
# Get the length of the tweet with the maximum number of tokens
max_tokens = int(len((x_train + x_test_tokens)[np.argmax(num_tokens)]))

# Zero is added before the values given in the padding operation.

x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens)

embedding_matrix = cnn_u.get_word2vec_embedding_matrix(100, tokenizer, filtered_words_list)
# on each row there is the vectorial representation for a word in out vocabulary

tf.compat.v1.disable_eager_execution()

# CNN
num_classes = 2
print(embedding_matrix.shape)
tf.compat.v1.enable_eager_execution()

# In[]:
# a layer instance is callable on a tensor, and returns a tensor
np.random.seed(0)  # Set a random seed for reproducibility

# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
main_input = Input(shape=(max_tokens,), dtype='int32', name='main_input')

# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
x = Embedding(output_dim=embedding_matrix.shape[1], input_dim=len(list(tokenizer.word_index)) + 1,
              input_length=max_tokens)(main_input)

lstm_out = LSTM(32)(x)
auxiliary_input = Input(shape=(3,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
main_output = Dense(1, activation='sigmoid', name='main_output')(x)
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])
model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy'})
# And trained it via:
model.fit({'main_input': np.array(x_train_pad), 'aux_input': np.array(sentiments_train)},
          {'main_output': np.array(y_train)},
          epochs=3, batch_size=100)
model.save("../data/model_1")
result = model.evaluate({'main_input': np.array(x_test_pad), 'aux_input': np.array(sentiments_test)},
                        {'main_output': np.array(y_test)},
                        batch_size=100,
                        verbose=1)
# pred = model.predict({'main_input': np.array(x_test_pad), 'aux_input': np.array(sentiments_test)})
print(result)


# In[]:
