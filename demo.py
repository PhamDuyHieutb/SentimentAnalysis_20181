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

    doc_clean = clean.clean_review(document)
    doc_clean_process_number = clean.number_process(doc_clean)
    print(doc_clean_process_number)
    tfidf = processor.transform(np.array([doc_clean_process_number]))

    reload = load("model_test")
    predict = reload.predict(tfidf)

    if predict == "2":
        print("POSITIVE :)")
    elif predict == "1":
        print("NEUTRAL")
    else:
        print("NEGATIVE -_-")



predict("""
it's too expensive
""")