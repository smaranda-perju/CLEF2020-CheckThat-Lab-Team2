from utils.text_filter import TextFilter
import sys
import csv

csv.field_size_limit(2147483647)


class DataUtility:
    @staticmethod
    def load_data(file):
        rows = []
        with open(file, encoding="utf8") as csv_file:
            spam_reader = csv.reader(csv_file, quoting=csv.QUOTE_MINIMAL)
            for csv_row in spam_reader:
                rows.append(csv_row)
        return rows
   def convert_result(mat):
        row = []
        result = []
        for i in range(0, len(mat[0])):
            row[0] = mat[i][0]
            row[1] = mat[i][1]
            result.append(row)
        return result
    def export_mat(mat):
            with open('../data/result.csv', mode='w') as cv:
            write = csv.writer(cv)
            for index in range(len(mat)):
                write.writerow([mat[index][0], mat[index][1]])
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
    def separate_unlabeled_data(in_data_text, in_data_label, in_sentiments):
        aux_list = []
        aux_sentiments = []
        aux_label_list = []
        unlabeled = []
        unlabeled_sentiments = []
        for i in range(len(in_data_text)):
            if in_data_label[i] != '':
                aux_list.append(in_data_text[i])
                aux_label_list.append(in_data_label[i])
                aux_sentiments.append(in_sentiments[i])
            else:
                unlabeled.append(in_data_text[i])
                unlabeled_sentiments.append(in_sentiments[i])
        return aux_list, aux_label_list, aux_sentiments, unlabeled, unlabeled_sentiments

    @staticmethod
    def convert_labels_to_int(labels_list):
        int_labels = []
        for i in labels_list:
            int_labels.append(int(i))
        print(int_labels)
        return int_labels
