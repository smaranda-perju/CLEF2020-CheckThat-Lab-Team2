import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import csv
import nltk
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow_core.python.keras.models import load_model

from utility.data_utility import DataUtility as du
from utility.cnn_utils import CNNUtils as cnn_u

nltk.download('stopwords')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# In[98]:

# Load data
tweets = du.load_data('../data/test_dataset.csv')
news = du.load_data("../data/fake_news_dataset.csv")

# Remove headers
du.remove_data_header(tweets)
du.remove_data_header(news)

# Get only cells with tweets text and labels
tweets_text_list, tweets_label_list = du.get_selected_columns(tweets, 1, 8)
news_text_list, news_label_list = du.get_selected_columns(news, 3, 4)
for i in range(0, len(news_text_list)):
    tweets_text_list.append(news_text_list[i][:280])
    tweets_label_list.append(news_label_list[i])

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

x_train_pad = pad_sequences(x_train_tokens, maxlen=100)
x_test_pad = pad_sequences(x_test_tokens, maxlen=100)

embedding_matrix = cnn_u.get_word2vec_embedding_matrix(100, tokenizer, filtered_words_list)
# on each row there is the vectorial representation for a word in out vocabulary

tf.compat.v1.disable_eager_execution()

# CNN
num_classes = 2

# Model parameters
num_filters = 64
weight_decay = 1e-5

model, callbacks_list = cnn_u.build_model(tokenizer, embedding_matrix, 100, num_classes, num_filters,
                                          weight_decay)
# Training params
batch_size = 100
num_epochs = 10
history = model.fit(x_train_pad, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list,
                    validation_split=0.1, shuffle=True, verbose=2)
model.save('model_combined.h5')

# Evaluate the model
model.evaluate(x_test_pad, y_test, batch_size, verbose=1, sample_weight=None, steps=None, callbacks=callbacks_list,
               max_queue_size=10, workers=1, use_multiprocessing=False)

# Use the model on new data
x_unlabeled_tokens = tokenizer.texts_to_sequences(unlabeled_tweets)
x_unlabeled_pad = pad_sequences(x_unlabeled_tokens, maxlen=100)
result = model.predict(x_unlabeled_pad, batch_size, verbose=0, steps=None, callbacks=callbacks_list, max_queue_size=10,
                       workers=1, use_multiprocessing=False)
with open('../data/result.csv', mode='w') as cv:
    write = csv.writer(cv)
    for index in range(len(result)):
        write.writerow([result[index][0], result[index][1]])

# In[]:

# Comparand cum se comporta modelul antrenat pe fake news pe tweeturi

model_tweets = load_model('model_tweets.h5')
model_fake_news = load_model('model_fake_news.h5')
model_combined = load_model('model_combined.h5')

# In[]:

model_combined.evaluate(x_test_pad, y_test, batch_size, verbose=1, sample_weight=None, steps=None,
                        callbacks=callbacks_list,
                        max_queue_size=10, workers=1, use_multiprocessing=False)

# In[]:
