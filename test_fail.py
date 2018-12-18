from fileio import FileIO
from nltk import word_tokenize
from preprocess import Clean

io = FileIO()
clean = Clean()

# get reviews failed to judge reasons

def make_label(stars):
    if int(float(stars)) > 3:
        return 2     # positive
    elif int(float(stars)) < 3:
        return 0     # negative
    else:
        return 1     # neutral

def write_tuple_data(data, path_write):
    corpus = []
    for label, revi in data:
        corpus.append(str(label) + "--" + revi)
    io.write_file_text('\n'.join(corpus), path_write)

def filterTestDataByDict(path_file, path_write_raw, path_write_clean, path_dictionary):
    """
    :return: filter data by dictionary
    """
    dictionary = io.read_file_text(path_dictionary).split(", ")

    corpus = []
    corpus_raw = []
    corpus_clean = []
    data_processed = io.read_file_json(path_file)

    for re in data_processed:
        corpus.append((re['rating'], re['review_body']))

    for star, revi in corpus:
        label = make_label(star)
        rev_clean = clean.number_process(clean.negation_process(clean.clean_review(revi), 'negation'))
        arr_word  = word_tokenize(rev_clean)
        elements_in_both_lists = [w for w in arr_word if w in dictionary]
        if len(elements_in_both_lists) > 0:
            corpus_raw.append((label, revi))
            corpus_clean.append((label, rev_clean))

    write_tuple_data(corpus_raw, path_write_raw)
    write_tuple_data(corpus_clean, path_write_clean)


def get_index_fail(predict, label):
    count = -1
    index_fails = []
    for p in predict:
        count += 1
        if p != label[count]:
            index_fails.append(str(count)+ ' ' + str(p) + ' ' + str(label[count]))
    return index_fails

def get_reviews_fail(indexs, corpus_raw, corpus_clean):
    list_reviews_fail = []
    for i in indexs:
        index = int(i.split(" ")[0])
        list_reviews_fail.append(i + ' - ' + corpus_raw[index] + " - " + corpus_clean[index])
    io.write_file_text('\n'.join(list_reviews_fail), 'data/list_reviews_fail')


def process_test_data_for_fail_reviews():

    data_raw = io.read_file_json("data/data_raw/test_raw/test.json")
    corpus = []
    for i in data_raw:
        corpus.append(clean.number_process(clean.negation_process(clean.clean_review(i['review_body']), 'negation')))

    labels = [x['rating'] for x in data_raw]
    # write data to txt file to get review fail predicted
    last_data = list(zip(labels, corpus))
    with open('data/test_fail_review', 'w') as fp:
        fp.write('\n'.join('%s -- %s' % x for x in last_data))


Y_test = io.read_file_text("data/datatestsvm_label1").split("\n")

#filterTestDataByDict("data/data_raw/test_raw/test.json", "data/result_test/test_raw", "data/result_test/test_clean",'dictionary')
predict = io.read_file_text('data/predict_label').split("\n")
index_fail = get_index_fail(predict, Y_test)

corpus_raw = io.read_file_text("data/result_test/test_raw").split("\n")
corpus_clean = io.read_file_text("data/result_test/test_clean").split("\n")

get_reviews_fail(index_fail, corpus_raw, corpus_clean)



