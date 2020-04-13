import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import csv
import nltk
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

from data_utility import DataUtility as du
from cnn_utils import CNNUtils as cnn_u

nltk.download('stopwords')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# In[98]:

# Load data
tweets = du.load_data('test_dataset.csv')
# Remove headers
du.remove_data_header(tweets)
# Get only cells with tweets text and labels
tweets_text_list, tweets_label_list = du.get_selected_columns(tweets, 1, 8)
# Filter data
filtered_tweets, filtered_words_list = du.filter_data(tweets_text_list)

# Eliminate the tweets that don't have a label
filtered_tweets, tweets_label_list, unlabeled_tweets = du.separate_unlabeled_data(filtered_tweets, tweets_label_list)
# Convert labels from string to int
tweets_label_list = du.convert_labels_to_int(tweets_label_list)

# Split data into train and test
df = pd.DataFrame({"tweet": filtered_tweets, "label": tweets_label_list})
target = df['label'].values.tolist()
data = df['tweet'].values.tolist()
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)

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

# Model parameters
num_filters = 64
weight_decay = 1e-5

model, callbacks_list = cnn_u.build_model(tokenizer, embedding_matrix, max_tokens, num_classes, num_filters,
                                          weight_decay)
# Training params
batch_size = 100
num_epochs = 10
model.fit(x_train_pad, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list,
          validation_split=0.1, shuffle=True, verbose=2)
model.save('model1.h5')

# Evaluate the model
model.evaluate(x_test_pad, y_test, batch_size, verbose=1, sample_weight=None, steps=None, callbacks=callbacks_list,
               max_queue_size=10, workers=1, use_multiprocessing=False)

# Use the model on new data
x_unlabeled_tokens = tokenizer.texts_to_sequences(unlabeled_tweets)
x_unlabeled_pad = pad_sequences(x_unlabeled_tokens, maxlen=max_tokens)
result = model.predict(x_unlabeled_pad, batch_size, verbose=0, steps=None, callbacks=callbacks_list, max_queue_size=10,
                       workers=1, use_multiprocessing=False)
with open('result.csv', mode='w') as cv:
    write = csv.writer(cv)
    for index in range(len(result)):
        write.writerow([result[index][0], result[index][1]])
