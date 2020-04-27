import numpy as np
from gensim.models import Word2Vec


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
