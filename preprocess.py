'''
divide data raw to train and test part
make bagofwords
filter data with that bagofwords
convert data to form of libsvm
'''
import os
import re
from fileio import FileIO
from nltk import word_tokenize
from dataprocessor import DataProcessor
from scipy import sparse
from random import shuffle
from sklearn.model_selection import train_test_split
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer


io = FileIO()

def make_label(stars):
    if int(float(stars)) > 3:
        return 2     # positive
    elif int(float(stars)) < 3:
        return 0     # negative
    else:
        return 1     # neutral

class Clean:


    def clean_review(self, review):

        lematizer = WordNetLemmatizer()
        stemmer = LancasterStemmer()

        match = {
            'one'  : '1',
            'two'  : '2',
            'three': '3',
            'four' : '4',
            'five' : '5',
            ' not': 'n\'t',
            '': '(\'s|\'ll|\'re|\'d|\'ve)',
            ' ': '[^a-zA-Z]'
        }

        for key in match:
            review = re.sub(match[key], key, review)

        token_review = word_tokenize(review)
        # remove stopwords
        stops = io.read_file_text("stopwords").strip().split("\n")
        meaningful_words = [w for w in token_review if w.lower() not in stops]
        stem = []
        # negation process
        listwords_negation_processed = self.negation_process(meaningful_words, "negation")

        for w in listwords_negation_processed:
            st = stemmer.stem(lematizer.lemmatize(w.lower(), pos = 'v'))
            stem.append(st)

        return " ".join(stem)


    def negation_process(self, listwords, path_negation_words):

        """
        concat not_ with next word of negation words
        """

        list_negation = io.read_file_text(path_negation_words).split("\n")
        list_words_after = []
        temp_word = ''
        for w in listwords:
            if w in list_negation and temp_word != "not":
                temp_word = "not"
            else:
                if temp_word == "not":
                    w = "not_" + w
                    temp_word = ''
                list_words_after.append(w)

        return list_words_after


    def number_process(self, review):
        """
        :param review: input single review
        :return: review which was handled about number star (five star => five_star)
        """
        list_number = ['one', 'two', 'three', 'four', 'fiv']
        list_words_before = word_tokenize(review)
        list_words_after = []
        temp_word = ''
        for w in list_words_before:
            if w in list_number:
                temp_word = w
            else:
                if temp_word in list_number and w == "star":    # two star => two_star
                    w = temp_word + '_' + w
                    list_words_after.append(w)
                else:
                    list_words_after.append(w)
                    list_words_after.append(temp_word)

                temp_word = ''

        return " ".join(list_words_after)


clean = Clean()


def preprocess_corpus(path_data, output_processed_negation_number):
    """
    clean data and negation handling
    """

    data = io.read_file_json(path_data)
    for re in data:
        content = clean.clean_review(re['review_body'].strip())
        content_num_proces = clean.number_process(content)
        re['review_body'] = content_num_proces

    io.write_file_json(data, output_processed_negation_number)

def make_bag_of_words(path_traindata_processed):

    '''
    :return: bag of words after filter redundant
    '''

    all_word = []
    words = []

    data_processed = io.read_file_json(path_traindata_processed)

    # for with each reviews in each file => add to words[]
    for i in data_processed:
        content = str(i['review_body']).strip().split(' ')
        words.append(content)

    for i in range(len(words)):  # for each reviews => count frequence of each word
        list_words = list(set(words[i]))     # distict list word in a review

        freq = {}
        for word in list_words:
            freq[word] = 0  # gan tan so ban dau cho cac tu bang 0

        for j in range(len(words[i])):  # test lai
            freq[words[i][j]] += 1     # tinh tan so cua cac tu
        review = [w for w in list_words if freq[w] > 1 and freq[w] < 25000]   # remove words which has frequent <= 4

        new_line = []

        for word in review:
            new_line.append(word)
            all_word += new_line

    bagOfWords = set(all_word)
    print("leng dict before " , len(bagOfWords))

    listwordremoved = []
    for word in bagOfWords:
        count = 0
        for re in words:
            for w in re:
                if word == w:
                    count += 1
                    break
        if count > 35000 or count < 3:
            listwordremoved.append(word)
    for w in listwordremoved:
        bagOfWords.remove(w)

    print("length dict after ", len(bagOfWords))
    io.write_file_text('\n'.join(listwordremoved), 'words_removed')
    io.write_file_text(', '.join(bagOfWords), 'bagofwords')

def filterTestDataByBagOfWords(path_file, path_write_raw, path_write_clean, path_bagofwords):
    """
    :return: filter data by bag of words
    """
    bagOfWords = io.read_file_text(path_bagofwords).split(", ")

    corpus = []
    corpus_raw = []
    corpus_clean = []
    data_processed = io.read_file_json(path_file)

    for re in data_processed:
        corpus.append((re['rating'], re['review_body']))

    for star, revi in corpus:
        label = make_label(star)
        rev_clean = clean.number_process(clean.clean_review(revi))
        arr_word  = word_tokenize(rev_clean)
        elements_in_both_lists = [w for w in arr_word if w in bagOfWords]
        if len(elements_in_both_lists) > 0:
            corpus_raw.append((label, revi))
            corpus_clean.append((label, rev_clean))

    io.write_tuple_data(corpus_raw, path_write_raw)
    io.write_tuple_data(corpus_clean, path_write_clean)


def filterTrainDataByBagOfWordsAndClassify(path_file, path_write, path_bagofwords):

    """
    filter train data by bag of words and classify to 3 label 0, 1, 2
    """

    bagOfWords = io.read_file_text(path_bagofwords).split(", ")
    data_0 = []
    data_1 = []
    data_2 = []
    data_processed = io.read_file_json(path_file)
    for revi in data_processed:
        arr_text = str(revi['review_body']).strip().split(' ')
        elements_in_both_lists = [w for w in arr_text if w in bagOfWords]
        label = make_label(revi['rating'])
        if len(elements_in_both_lists) > 0:
            if label == 2:
                data_2.append(" ".join(elements_in_both_lists))
            elif label == 0:
                data_0.append(" ".join(elements_in_both_lists))
            else:
                data_1.append(" ".join(elements_in_both_lists))

    print("length of data " + str(len(data_0 + data_1 + data_2)))

    io.write_file_text('\n'.join(data_0), path_write + "/label_0")
    io.write_file_text('\n'.join(data_1), path_write + "/label_1")
    io.write_file_text('\n'.join(data_2), path_write + "/label_2")


dataprocessor = DataProcessor()

def transformToTfidf(path_data, type_data):

    corpus = []
    labels  = []
    # read file and split label, review

    if type_data == 'test':
        data = io.read_file_text(path_data).split("\n")
        for i in data:
            split_data = i.split("--")
            label = split_data[0]
            review = split_data[1]
            labels.append(label)
            corpus.append(review)
        tfidf = dataprocessor.transform(corpus)

    else:
        list_files = os.listdir(path_data)
        for file_name in list_files:
            label = file_name.split("_")[1]
            dataByLabel = io.read_file_text(path_data + "/" + str(file_name)).split("\n")
            for revi in dataByLabel:
                labels.append(label)
                corpus.append(revi)
        tfidf = dataprocessor.fit(corpus)

    # transform data to tfidf with option for training and testing

    return tfidf, labels

# def convert(input, output):
#     all_data = []
#     for revi, label in input:
#         label_revi = str(label)
#         index = -1
#         for tfidf in revi:
#             index += 1
#             if tfidf != 0:
#                 feature = str(index) + ":" + str(tfidf)
#                 label_revi = label_revi + " " + feature
#         if len(label_revi.split(" ")) > 1:   # has min 1 feature
#             all_data.append(label_revi)
#     io.write_file_text('\n'.join(all_data), output)

def convertDataToFormOfSVM(path_file, output_revi, output_label, type):

    """
    transform data to tfidf and map to sparse vector
    """

    revi, label = transformToTfidf(path_file,type)
    sparse.save_npz(output_revi, sparse.csr_matrix(revi))
    io.write_file_text('\n'.join(label), output_label)

def split_train_test(path_data, output_train, output_test):
    data = io.read_file_json(path_data)
    shuffle(data)
    train_data, test_data = train_test_split(data, test_size= 0.2, random_state= 42)
    io.write_file_json(train_data, output_train)
    io.write_file_json(test_data, output_test)


def balanceReviews(path_all_data, path_balance_data):

    """
    balace num of reviews in each label
    """

    data = io.read_multi_files_json(path_all_data)
    data_sample = []
    count = 0
    index = 0
    for lb in [0,1,2]:
        if lb == 0:
            while count < 25000:
                re = data[index]
                label = make_label(re['rating'])
                if (label == lb):
                    data_sample.append(re)
                    count += 1
                index += 1

        elif lb == 1:
            while count < 25000:
                re = data[index]
                label = make_label(re['rating'])
                if (label == lb):
                    data_sample.append(re)
                    count += 1
                index += 1
        else:
            while count < 25000:
                re = data[index]
                label = make_label(re['rating'])
                if (label == lb):
                    data_sample.append(re)
                    count += 1
                index += 1
        index = 0
        count = 0

    print(len(data_sample))
    io.write_file_json(data_sample, path_balance_data)

def main():

    balanceReviews("/home/hieupd/PycharmProjects/data_for_sentiment/electronic","data/data_raw/elec_balance.json")

    # split train test before clean
    print("split")
    split_train_test("data/data_raw/elec_balance.json", "data/data_raw/train.json", "data/data_raw/test.json")

    # preprocess train and test data
    print("preprocess")
    preprocess_corpus("data/data_raw/train.json","data/data_processed/elec_clean_train.json")
    make_bag_of_words("data/data_processed/elec_clean_train.json")
    #
    print("filter")
    filterTrainDataByBagOfWordsAndClassify("data/data_processed/elec_clean_train.json", "data/data_filtered_by_dict/train", "bagofwords")
    filterTestDataByBagOfWords("data/data_raw/test.json", "data/data_raw/test_raw_for_fail_check", "data/data_filtered_by_dict/test_filter", "bagofwords")
    #
    print("convert to svm form")
    convertDataToFormOfSVM("data/data_filtered_by_dict/train", "data/datatrainsvm1.npz",  "data/datatrainsvm_label1",'train')
    convertDataToFormOfSVM("data/data_filtered_by_dict/test_filter", "data/datatestsvm1.npz", 'data/datatestsvm_label1', 'test')


if __name__ == '__main__':
    main()