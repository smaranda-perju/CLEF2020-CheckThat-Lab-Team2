import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import csv
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, \
    GlobalMaxPooling1D
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Data load

tweets = []


with open('test_dataset.csv', encoding="utf8") as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        tweets.append(row);
len(tweets)
tweets[1][1]

# In[95]:

tweets_text_list = []
tweets_label_list = []
for row in tweets:
    tweets_text_list.append(row[1])
    tweets_label_list.append(row[8])

# In[96]:


tweets_text_list
del tweets_text_list[0]

# In[97]:


del tweets_label_list[0]
tweets_label_list

# In[98]:


print("Tweets : ", len(tweets_label_list))
value1 = [i for i in tweets_label_list if i in '0']
value2 = [i for i in tweets_label_list if i in '1']
value3 = [i for i in tweets_label_list if i in '']
print("class 0: ", len(value1))
print("class 1: ", len(value2))
print("fara label: ", len(value3))

# In[99]:


aux_list = []
aux_label_list = []
unlabeled_tweets = []
# Trebuie sa eliminaaam tweet urile fara label
for i in range(len(tweets_text_list)):
    if tweets_label_list[i] != '':
        aux_list.append(tweets_text_list[i])
        aux_label_list.append(tweets_label_list[i])
    else:
        unlabeled_tweets.append(tweets_text_list[i])

# In[100]:


tweets_text_list = aux_list
tweets_label_list = aux_label_list
unlabeled_tweets

# In[101]:


print("Tweets : ", len(tweets_label_list))
value1 = [i for i in tweets_label_list if i in '0']
value2 = [i for i in tweets_label_list if i in '1']
value3 = [i for i in tweets_label_list if i in '']
print("class 0: ", len(value1))
print("class 1: ", len(value2))
print("fara label: ", len(value3))

# In[102]:


# Preprocesaaaaare


# In[103]:


from nltk.stem import PorterStemmer
import re


def splitIntoStem(message):
    return [removeNumeric(stripEmoji(singleCharacterRemove(removePunctuation
                                                           (removeHyperlinks
                                                            (removeHashtags
                                                             (removeUsernames
                                                              (stemWord(word)))))))) for word in message.split()]


def stemWord(tweet):
    ps = PorterStemmer()
    return ps.stem(tweet).lower()


# Remove usernames
def removeUsernames(tweet):
    return re.sub('@[^\s]+', '', tweet)


# Remove hashtag
def removeHashtags(tweet):
    return re.sub(r'#[^\s]+', '', tweet)


# Remove link
def removeHyperlinks(tweet):
    return re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', tweet)


# Remove numeric character
def removeNumeric(value):
    blist2 = [item for item in value if not item.isdigit()]
    blist3 = "".join(blist2)
    return blist3


# Remove punctuation
def removePunctuation(tweet):
    return re.sub(r'[^\w\s]', '', tweet)


# Remove single character
def singleCharacterRemove(tweet):
    tweet = tweet.replace('_', '')
    return re.sub(r'(?:^| )\w(?:$| \ )', ' ', tweet)


# Remove emoji
def stripEmoji(text):
    RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
    return RE_EMOJI.sub(r'', text)


# In[104]:


filtered_tweets = []
unlabeled_filtered_tweets = []
for i, k in enumerate(tweets_text_list):
    filtered_tweets.append(" ".join(splitIntoStem(k)).split())
for i, k in enumerate(unlabeled_tweets):
    unlabeled_filtered_tweets.append(" ".join(splitIntoStem(k)).split())

# In[105]:


filtered_tweets[0]

# In[106]:


tweets_text_list[0]

# In[107]:


unlabeled_filtered_tweets[0]

# In[108]:


import nltk

nltk.download('stopwords')

# In[109]:


# Eliminaaam stop words


# In[110]:


def removeStopWords(tweet_list):
    filtered_stopwords = []
    filtered_stopwords_list = []

    stop_words = stopwords.words('english')

    print("stop_words : ", stop_words)

    for i in filtered_tweets:
        filtered_sentence = [w for w in i if not w in stop_words]
        filtered_stopwords_list.append(filtered_sentence)  # return list value
        filtered_stopwords.append(" ".join(filtered_sentence))  # return string value

    return filtered_stopwords, filtered_stopwords_list


filtered_tweets, filtered_stopwords_list = removeStopWords(filtered_tweets)

# In[111]:


filtered_tweets[0]

# In[112]:


unlabeled_filtered_tweets[0]

# In[113]:

def removeStopWords_u(tweet_list):
    filtered_stopwords_u = []
    filtered_stopwords_list_u = []

    stop_words = stopwords.words('english')

    print("stop_words : ", stop_words)

    for i in unlabeled_filtered_tweets:
        filtered_sentence = [w for w in i if not w in stop_words]
        filtered_stopwords_list_u.append(filtered_sentence)  # return list value
        filtered_stopwords_u.append(" ".join(filtered_sentence))  # return string value

    return filtered_stopwords_u, filtered_stopwords_list_u


filtered_unlabeled_tweets, filtered_stopwords_list = removeStopWords_u(unlabeled_filtered_tweets)

# In[114]:


filtered_unlabeled_tweets.remove('')
filtered_unlabeled_tweets

# In[115]:


tweets_text_list[0]

# In[116]:


df = pd.DataFrame({"tweet": filtered_tweets, "label": tweets_label_list})
df.head()

# In[117]:


df.iloc[0]

# In[118]:


target = df['label'].values.tolist()
data = df['tweet'].values.tolist()

# In[119]:


x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# In[120]:


print("x_train :", len(x_train))
print("x_test :", len(x_test))

# In[121]:


x_train[800]

# In[122]:


tokenizer = Tokenizer()  # Tokenizer(num_words=5000) => 5000 words of the highest frequency
tokenizer.fit_on_texts(data)
tokenizer

# In[123]:


list(tokenizer.word_index)[0:10]

# In[124]:


print("len(tokenizer) :", len(list(tokenizer.word_index)))

# In[125]:


# Training set token with text_to_sequences.

x_train_tokens = tokenizer.texts_to_sequences(x_train)

# In[126]:


x_train[800]

# In[127]:


print(x_train_tokens[800])

# In[128]:


x_test_tokens = tokenizer.texts_to_sequences(x_test)

# In[129]:


num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)

# In[130]:


num_tokens[800]

# In[131]:


np.max(num_tokens)

# In[132]:


np.argmax(num_tokens)

# In[133]:


x_train[np.argmax(num_tokens)]

# In[134]:


max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
max_tokens

# In[135]:


# How many of the eighteen length tweets are included?

np.sum(num_tokens < max_tokens) / len(num_tokens)

# In[136]:


x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens)

# In[137]:


# Zero is added before the values given in the padding operation.

print("x_train_tokens :", x_train_tokens[0])
print("x_train_pad :", x_train_pad[0])

# In[138]:


x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens)

# In[139]:


print("x_test_tokens :", x_test_tokens[0])
print("x_train_pad :", x_test_pad[0])

# In[140]:


print("x_train_pad.shape :", x_train_pad.shape)
print("x_train_pad.shape :", x_test_pad.shape)

# In[141]:


idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(), idx.keys()))


# In[142]:


def tokens_to_string(tokens):
    words = [inverse_map[token] for token in tokens if token != 0]
    text = ' '.join(words)
    return text


# In[143]:


x_train[800]

# In[144]:


print(x_train_tokens[800])

# In[145]:


tokens_to_string(x_train_tokens[800])

# In[146]:


# Embedding operations


# In[147]:


filtered_stopwords_list[0]

# In[150]:


from gensim.models import Word2Vec

model = Word2Vec(filtered_stopwords_list, min_count=1, size=100, workers=3, window=3, sg=1)
print(model)
model_save_location = "tweets_notbinary"
model.wv.save_word2vec_format(model_save_location)

# In[151]:


model.wv.__getitem__(filtered_stopwords_list[1])

# In[152]:


embedding_size = 100

# In[153]:


# Word2vec load(2.option) example

word2vec = {}
with open('tweets_notbinary', encoding='UTF-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        print(word)
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec

# In[154]:


print("x_test[1] :", x_test[1])
print("x_test_pad[1] :", x_test_pad[1])

# In[155]:


num_words = len(list(tokenizer.word_index))
num_words

# In[156]:


embedding_matrix = np.random.uniform(-1, 0, (num_words + 1, embedding_size))
for word, i in tokenizer.word_index.items():
    if i < num_words:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# In[182]:


embedding_matrix.shape

# In[183]:


word2vec

# In[181]:


embedding_matrix[1]

# In[160]:


sequence_length = max_tokens
vocabulary_size = num_words
embedding_dim = embedding_size
filter_sizes = [3, 4, 5]
num_filters = 512
drop = 0.5
epochs = 5
batch_size = 30

# In[161]:


y_train2 = []
y_test2 = []
for i in y_train:
    y_train2.append(int(i))
for i in y_test:
    y_test2.append(int(i))

# In[162]:


tf.compat.v1.disable_eager_execution()

# In[163]:


# CNN architecture

num_classes = 2

# Training params
batch_size = 256
num_epochs = 5

# Model parameters
num_filters = 64
embed_dim = embedding_size
weight_decay = 1e-5

print("training CNN ...")
model = Sequential()

# Model add word2vec embedding

model.add(Embedding(input_dim=num_words + 1,
                    output_dim=embedding_size,
                    weights=[embedding_matrix],
                    input_length=max_tokens,
                    trainable=True,  # the layer is trained
                    name='embedding_layer'))
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dense(num_classes, activation='softmax'))  # multi-label (k-hot encoding)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

# define callbacks
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]

hist = model.fit(x_train_pad, y_train2, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list,
                 validation_split=0.1, shuffle=True, verbose=2)

# In[164]:


model.fit(x_train_pad, y_train2, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list,
          validation_split=0.1, shuffle=True, verbose=2)

# In[165]:


model.save('model1.h5')

# In[166]:


# Incercaaaaam sa evaluuuaaaam modelul


# In[167]:


x_test_pad

# In[168]:


num_classes = 2

# Training params
batch_size = 256
num_epochs = 5

# Model parameters
num_filters = 64  # görüntünün boyutu mesela 512*512
embed_dim = embedding_size
weight_decay = 1e-5

# In[169]:


model.evaluate(x_test_pad, y_test2, batch_size, verbose=1, sample_weight=None, steps=None, callbacks=callbacks_list,
               max_queue_size=10, workers=1, use_multiprocessing=False)

# In[170]:


result = model.predict(x_train_pad, batch_size, verbose=0, steps=None, callbacks=callbacks_list, max_queue_size=10,
                       workers=1, use_multiprocessing=False)

# In[172]:


result[3]

# In[173]:


y_train2[1]

# In[174]:


###Exportare rezultat ca csv
with open('rezultaaaat.csv', mode='w') as cv:
    scris = csv.writer(cv)
    for index in range(len(result)):
        scris.writerow([result[index][0], result[index][1], y_train[index]])

# In[175]:


x_train_pad[0]

# In[176]:


x_unlabeled_tokens = tokenizer.texts_to_sequences(filtered_unlabeled_tweets)
x_unlabeled_pad = pad_sequences(x_unlabeled_tokens, maxlen=max_tokens)

# In[178]:


# In[179]:


result = model.predict(x_train_pad, batch_size, verbose=0, steps=None, callbacks=callbacks_list, max_queue_size=10,
                       workers=1, use_multiprocessing=False)

# In[180]:


result

# In[ ]:
