from dataprocessor import DataProcessor
from preprocess import clean
from fileio import FileIO
from joblib import load
import numpy as np



processor = DataProcessor()
io = FileIO()

path_negation_words = 'negation'

def predict(document):

    # predict document input

    doc_clean = clean.negation_process(clean.clean_review(document),path_negation_words)
    doc_clean_process_number = clean.number_process(doc_clean)
    print(doc_clean_process_number)
    tfidf = processor.transform(np.array([doc_clean_process_number]))

    reload = load("model_test")
    predict = reload.predict(tfidf)

    print(predict)

predict("Wonderful promotion for Galaxy of Heroes. Smooth transaction.")