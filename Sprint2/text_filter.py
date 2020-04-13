import re

from gensim.parsing import PorterStemmer
from nltk.corpus import stopwords


class TextFilter:

    @staticmethod
    def split_into_stem(message):
        return [TextFilter.remove_numeric(
            TextFilter.remove_emoji(
                TextFilter.remove_single_character(
                    TextFilter.remove_punctuation(
                        TextFilter.remove_hyperlinks(
                            TextFilter.remove_hashtags(
                                TextFilter.remove_usernames(
                                    TextFilter.stem_word(in_word)))))))) for in_word in message.split()]

    @staticmethod
    def stem_word(in_word):
        ps = PorterStemmer()
        return ps.stem(in_word).lower()

    @staticmethod
    def remove_usernames(in_word):
        return re.sub('@[^\s]+', '', in_word)

    @staticmethod
    def remove_hashtags(in_word):
        return re.sub(r'#[^\s]+', '', in_word)

    @staticmethod
    def remove_hyperlinks(in_word):
        return re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', in_word)

    @staticmethod
    def remove_numeric(in_word):
        sub_string = [item for item in in_word if not item.isdigit()]
        sub_string = "".join(sub_string)
        return sub_string

    @staticmethod
    def remove_punctuation(in_word):
        return re.sub(r'[^\w\s]', '', in_word)

    @staticmethod
    def remove_single_character(in_word):
        in_word = in_word.replace('_', '')
        return re.sub(r'(?:^| )\w(?:$| \ )', ' ', in_word)

    @staticmethod
    def remove_emoji(text):
        re_emoji = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
        return re_emoji.sub(r'', text)

    @staticmethod
    def remove_stopwords(words_list):
        filtered_stopwords = []
        filtered_stopwords_list = []
        stop_words = stopwords.words('english')
        for ii in words_list:
            filtered_sentence = [w for w in ii if w not in stop_words]
            filtered_stopwords_list.append(filtered_sentence)  # return list value
            filtered_stopwords.append(" ".join(filtered_sentence))  # return string value
        return filtered_stopwords, filtered_stopwords_list
