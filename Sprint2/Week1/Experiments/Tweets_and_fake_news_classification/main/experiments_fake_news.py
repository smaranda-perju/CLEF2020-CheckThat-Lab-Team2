import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import csv
import nltk
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

from utility.data_utility import DataUtility as du
from utility.cnn_utils import CNNUtils as cnn_u

nltk.download('stopwords')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# In[98]:

# Load data
news = du.load_data("../data/fake_news_dataset.csv")
print(len(news))
# Remove headers
# for i in range(15000):
du.remove_data_header(news)

# In[]
# Get only cells with tweets text and labels
news_text_list, news_label_list = du.get_selected_columns(news, 3, 4)

print(news_text_list[0])
# Filter data
filtered_news, filtered_words_list = du.filter_data(news_text_list)

# In[]
# Convert labels from string to int
news_label_list = du.convert_labels_to_int(news_label_list)

# In[]
# Split data into train and test

df = pd.DataFrame({"news": filtered_news, "label": news_label_list})
target = df['label'].values.tolist()
data = df['news'].values.tolist()
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)

# In[]

# Training set token with text_to_sequences.

x_train_tokens = tokenizer.texts_to_sequences(x_train)
x_test_tokens = tokenizer.texts_to_sequences(x_test)

# In[]
num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)
# Get the length of the tweet with the maximum number of tokens
max_tokens = int(len((x_train + x_test_tokens)[np.argmax(num_tokens)]))
print(max_tokens)
# In[]
# Zero is added before the values given in the padding operation.
x_train_pad = pad_sequences(x_train_tokens, maxlen=100)
x_test_pad = pad_sequences(x_test_tokens, maxlen=100)

# In[]
embedding_matrix = cnn_u.get_word2vec_embedding_matrix(100, tokenizer, filtered_words_list)
# on each row there is the vectorial representation for a word in out vocabulary


# In[]

print(x_train_pad.shape)
print(y_train)
print(embedding_matrix.shape)
# In[]

tf.compat.v1.disable_eager_execution()

# CNN
num_classes = 2

# Model parameters
num_filters = 64
weight_decay = 1e-5

model, callbacks_list = cnn_u.build_model(tokenizer, embedding_matrix, 100, num_classes, num_filters,
                                          weight_decay)

# In[]
# Training params
batch_size = 100
num_epochs = 10
history = model.fit(x_train_pad, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list,
                    validation_split=0.1, shuffle=True, verbose=2)
#model.save('model_fake_news.h5')

# In[]

# Evaluate the model
model.evaluate(x_test_pad, y_test, batch_size, verbose=1, sample_weight=None, steps=None, callbacks=callbacks_list,
               max_queue_size=10, workers=1, use_multiprocessing=False)
