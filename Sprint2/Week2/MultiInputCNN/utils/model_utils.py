import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, \
    GlobalMaxPooling1D
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam


class CNNUtils:
    @staticmethod
    def tokens_to_string(tokens, p_tokenizer):
        inverse_map = dict(zip(p_tokenizer.word_index.values(), p_tokenizer.word_index.keys()))
        words = [inverse_map[token] for token in tokens if token != 0]
        text = ''.join(words)
        return text

    @staticmethod
    def get_word2vec_embedding_matrix(embedding_size, p_tokenizer, words_list):
        w2v_model = Word2Vec(words_list, min_count=1, size=embedding_size, workers=3, window=3, sg=1)
        model_save_location = "vocabulary_vec"
        w2v_model.wv.save_word2vec_format(model_save_location)
        word2vec = {}
        with open(model_save_location, encoding='UTF-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vec = np.asarray(values[1:], dtype='float32')
                word2vec[word] = vec
        words_number = len(list(p_tokenizer.word_index))
        result_embedding_matrix = np.random.uniform(-1, 0, (words_number + 1, embedding_size))
        for word, i in p_tokenizer.word_index.items():
            if i < words_number:
                embedding_vector = word2vec.get(word)
                if embedding_vector is not None:
                    result_embedding_matrix[i] = embedding_vector
        return result_embedding_matrix

    @staticmethod
    def build_model(p_tokenizer, p_embedding_matrix, p_max_tokens, number_of_classes, number_of_filters,
                    p_weight_decay):
        cnn_model = Sequential()
        cnn_model.add(Embedding(input_dim=len(list(p_tokenizer.word_index)) + 1,
                                output_dim=p_embedding_matrix.shape[1],
                                weights=[p_embedding_matrix],
                                input_length=p_max_tokens,
                                trainable=True,  # the layer is trained
                                name='embedding_layer'))
        cnn_model.add(Conv1D(number_of_filters, 7, activation='relu', padding='same'))
        cnn_model.add(MaxPooling1D(2))
        cnn_model.add(Conv1D(number_of_filters, 7, activation='relu', padding='same'))
        cnn_model.add(GlobalMaxPooling1D())
        cnn_model.add(Dropout(0.5))
        cnn_model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(p_weight_decay)))
        cnn_model.add(Dense(number_of_classes, activation='softmax'))  # multi-label (k-hot encoding)
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        cnn_model.summary()
        # define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
        return cnn_model, [early_stopping]
