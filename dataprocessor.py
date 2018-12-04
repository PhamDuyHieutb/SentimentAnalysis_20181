from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
import numpy as np
import os
from fileio import FileIO

io = FileIO()

class DataProcessor:

    def __init__(self) :
        self.__vectorizer = CountVectorizer()


    def fit(self, corpus) :
        vector = self.__vectorizer.fit_transform(corpus)
        tranform = TfidfTransformer()
        tfidf = tranform.fit_transform(vector.toarray())
        pickle.dump(self.__vectorizer.vocabulary_, open("feature.pkl", "wb"))
        return  tfidf.toarray()

    def transform(self, corpus):
        load_vec = CountVectorizer(vocabulary= pickle.load(open("feature.pkl", "rb")))

        tranform = TfidfTransformer()
        tfidf = tranform.fit_transform(load_vec.fit_transform(corpus))

        return tfidf.toarray()






