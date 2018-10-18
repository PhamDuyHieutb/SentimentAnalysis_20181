import json
import os
import DataProcessor
import math

PATHDATA_PROCESSED  =  "xxx"

def read_file_json(path_file):
    with open(path_file, 'r') as content:
        data = json.load(content)
    return data

def read_file_text(path_file):
    with open(path_file, 'r') as content:
        data = content.read()
    return data

def write_file(data, path_file):
    with open(path_file, 'w') as outfile:
        json.dump(data, outfile)

def make_label(stars):
    if int(stars) > 3:
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


def convertDataToFormOfSVM(path_data):
    data_processed = read_file_json(path_data)
    for i in data_processed:
        label = make_label(i['rating'])


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
        write_file(freq, "count_words")

        new_line = []
        for word in list_words:  # tinh tf-idf cho cac tu trong 1 review
            # value = tf_idf(word, words[i], words)
            # if value >= 2e-05:   # can test lai
            new_line.append(word)
            all_word += new_line

    dictionary = set(all_word)
    print("leng dict" , len(dictionary))
    with open("dictionary", 'w') as writs:
        writs.write(', '.join(dictionary))

    print(dictionary)
    # filter review with dictionary
    for file in list_files:
        temp = []
        path_file = path_data_processed + f"/{file}"
        data_processed = read_file_json(path_file)
        for revw in data_processed:
            arr_text = revw['review_body'].strip().split(' ')
            print(arr_text)
            elements_in_both_lists = [w for w in arr_text if w in dictionary]
            temp.append((revw['rating'], elements_in_both_lists))
        write_file(temp, f"/datatrain_processed/{file}")



def filterDataTestByDict(path_data_test,path_dict):
    dictionary = read_file_text(path_dict)
    list_files = os.listdir(path_data_test)
    for file in list_files:
        temp = []
        path_file = path_data_test + f"/{file}"
        data_processed = read_file_json(path_file)
        for revw in data_processed:
            arr_text = revw['review_body'].strip().split(' ')
            print(arr_text)
            elements_in_both_lists = [w for w in arr_text if w in dictionary]
            temp.append((revw['rating'], elements_in_both_lists))
        write_file(temp, f"/datatest_processed/{file}")

dataprocessor = DataProcessor()

