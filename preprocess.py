import json
import os
from DataProcessor import DataProcessor
import math
from random import shuffle

PATHDATA_PROCESSED  =  "xxx"

def read_file_json(path_file):
    with open(path_file, 'r') as content:
        data = json.load(content)
    content.close()
    return data

def read_file_text(path_file):
    with open(path_file, 'r') as content:
        data = content.read()
    content.close()
    return data

def read_file_text_by_lines(path_file):
    with open(path_file, 'r') as content:
        data = content.readlines()
    content.close()
    return data


def write_file_json(data, path_file):
    outfile = open(path_file, 'w')
    json.dump(data, outfile)
    outfile.close()

def write_file_text(data, path_file):
    out = open(path_file, 'w')
    out.write(data)
    out.close()


def make_label(stars):
    if int(float(stars)) > 3:
        return 1
    else:
        return 0


# caculate tf-idf
def tf_idf(word, current_doc, documents):
    current_count = 0
    current_len = len(current_doc)
    for w in current_doc:
        if word == w:
            current_count += 1
    tf = current_count / current_len

    all_count = 0
    all_len = len(documents)
    for d in documents:
        if word in d:
            all_count += 1
    if all_count == 0:
        all_count = 1
    idf = math.log(all_len / all_count, 10)

    return tf * idf

def make_dictionary(path_data_processed):
    all_word = []
    words = []
    list_files = os.listdir(path_data_processed)

    # for with each file in list file json by categories
    for file in list_files:
        path_file = path_data_processed + f"/{file}"
        data_processed = read_file_json(path_file)
        # for with each reviews in each file => add to words[]
        for i in data_processed:
            content = i['review_body'].strip().split(' ')
            words.append(content)

    for i in range(len(words)):  # for each reviews => count frequence of each word
        list_words = list(set(words[i]))     # distict list word in a review
        freq = {}
        for word in list_words:
            freq[word] = 0  # gan tan so ban dau cho cac tu bang 0

        for j in range(len(words[i])):
            freq[words[i][j]] += 1     # tinh tan so cua cac tu
        write_file_json(freq, "count_words")

        new_line = []
        for word in list_words:  # tinh tf-idf cho cac tu trong 1 review
            # value = tf_idf(word, words[i], words)
            # if value >= 2e-05:   # can test lai
            new_line.append(word)
            all_word += new_line

    dictionary = set(all_word)
    print("leng dict" , len(dictionary))
    write_file_text(', '.join(dictionary), 'dictionary')


def filterDataByDictAndClassify(path_data, path_write, path_dictionary):

    dictionary = read_file_text(path_dictionary)
    list_files = os.listdir(path_data)
    for label in range(0,2):
        data = []
        for file in list_files:
            path_file = path_data + f"/{file}"
            data_processed = read_file_json(path_file)
            for revi in data_processed:
                arr_text = revi['review_body'].strip().split(' ')
                elements_in_both_lists = [w for w in arr_text if w in dictionary]
                if make_label(revi['rating']) == label:
                    data.append(" ".join(elements_in_both_lists))
            write_file_text('\n'.join(data), path_write + f"/label_{label}")


dataprocessor = DataProcessor()

def transformToTfidf(path_data):

    corpus = []
    labels  = []


    list_files = os.listdir(path_data)
    for file_name in list_files:
        label = file_name.split("_")[1]
        dataByLabel = read_file_text_by_lines(path_data + f"/{file_name}")
        for revi in dataByLabel:
            labels.append(label)
            corpus.append(revi)
    tfidf = dataprocessor.fit(corpus)
    return tfidf, labels

def convert(input, output):
    all_data = []
    for revi, label in input:
        label_revi = str(label)
        index = -1
        for tfidf in revi:
            index += 1
            if tfidf != 0:
                feature = str(index) + ":" + str(tfidf)
                label_revi = label_revi + " " + feature
        if len(label_revi.split(" ")) > 3:
            all_data.append(label_revi)
    write_file_text('\n'.join(all_data), output)

def convertDataToFormOfSVM(path_data, output):
    revi, label = transformToTfidf(path_data)
    input_train = list(zip(revi, label))
    #shuffle(input_train)

    convert(input_train, output)



def main():
    make_dictionary("train")
    filterDataByDictAndClassify("train", "data_filtered_by_dict/train", "dictionary")
    filterDataByDictAndClassify("test", "data_filtered_by_dict/test", "dictionary")
    convertDataToFormOfSVM("data_filtered_by_dict/train", "datatrainsvm")
    #convertDataToFormOfSVM("data_filtered_by_dict/test", "datatestsvm")


if __name__ == '__main__':
    main()