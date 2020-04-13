import csv

from text_filter import TextFilter


class DataUtility:
    @staticmethod
    def load_data(file):
        rows = []
        with open(file, encoding="utf8") as csv_file:
            spam_reader = csv.reader(csv_file)
            for csv_row in spam_reader:
                rows.append(csv_row)
        return rows

    @staticmethod
    def get_selected_columns(in_data, col_x, col_y):
        x_list = []
        y_list = []
        for data_row in in_data:
            x_list.append(data_row[col_x])
            y_list.append(data_row[col_y])
        return x_list, y_list

    @staticmethod
    def remove_data_header(in_data):
        del in_data[0]

    @staticmethod
    def filter_data(words_list):
        filtered = []
        for ii, j in enumerate(words_list):
            filtered.append(" ".join(TextFilter.split_into_stem(j)).split())
        filtered, filtered_words_list = TextFilter.remove_stopwords(filtered)
        return filtered, filtered_words_list

    @staticmethod
    def separate_unlabeled_data(in_data_text, in_data_label):
        aux_list = []
        aux_label_list = []
        unlabeled = []
        for i in range(len(in_data_text)):
            if in_data_label[i] != '':
                aux_list.append(in_data_text[i])
                aux_label_list.append(in_data_label[i])
            else:
                unlabeled.append(in_data_text[i])
        return aux_list, aux_label_list, unlabeled

    @staticmethod
    def convert_labels_to_int(labels_list):
        int_labels = []
        for i in labels_list:
            int_labels.append(int(i))
        return int_labels
