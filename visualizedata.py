import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from joblib import load
from scipy import sparse
from fileio import FileIO
from sklearn.metrics import precision_score, recall_score, f1_score

io = FileIO()
# nhan lop
classes = [0,1,2]

# path data
DATATEST = "data/libsvm-3.22/datatestsvm"
DATATRAIN = "data/libsvm-3.22/datatrainsvm"
LABEL_PREDICT_PATH = 'data/libsvm-3.22/test/result'

X_train = sparse.load_npz("data/datatrainsvm1.npz")
Y_train = io.read_file_text("data/datatrainsvm_label1").split("\n")
X_test = sparse.load_npz("data/datatestsvm1.npz").toarray()
Y_test = io.read_file_text("data/datatestsvm_label1").split("\n")


def read_file(file_path):
    """
    Read file from disk
    file_path
    """
    file = open(file_path, 'r')
    try:
        text = file.read()
    except UnicodeDecodeError:
        print("fail open file: " + file_path)
        text = ''
    file.close()
    return text

def plot_confusion_matrix(cm,classes,normalize=False,cmap = plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm,interpolation= 'nearest',cmap=cmap)
    plt.title("Electronic")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation = 90)
    plt.yticks(tick_marks,classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if i == j:
            plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def main():

    # predict = io.read_file_text('data/predict_label').split("\n")
    # corpus_raw = []
    #
    # test_raw = io.read_file_json("data/data_raw/test_raw/test.json")
    # for re in test_raw:
    #     corpus_raw.append(re['review_body'])
    #
    # for i in range(0,11):
    #     re = corpus_raw[i]
    #     print(re)
    #
    #
    #
    # corpus_clean = io.read_file_text('data/test_fail_review').split("\n")
    # corpus_clean_sub = []
    # for re in corpus_clean:
    #     corpus_clean_sub.append(re['rating'] + " --" + re['review_body'])
    #
    # for i in range(0, 11):
    #     print(corpus_clean_sub[i])


#######################################

    # reload = load('model_test')
    # Y_predict = reload.predict(X_test)
    # io.write_file_text("\n".join(Y_predict), "data/predict_label"

    Y_predict = io.read_file_text("data/predict_label").split("\n")

    # print("score")
    # print(reload.score(X_test,Y_test))

    cfm = confusion_matrix(Y_test, Y_predict)
    print("recall " + str(recall_score(Y_test, Y_predict, average = "macro")))
    print("precision " + str(precision_score(Y_test,Y_predict, average = "macro")))
    print("F1 " + str(f1_score(Y_test,Y_predict, average = "macro")))


    plt.figure()
    plot_confusion_matrix(cfm, classes=classes, normalize=True)
    plt.show()


if __name__ == '__main__':
    main()


