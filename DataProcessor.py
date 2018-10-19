from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class DataProcessor () :

    def __init__(self) :
        self.__vectorizer = CountVectorizer()


    def fit(self, corpus) :
        vector = self.__vectorizer.fit_transform(corpus)
        tranform = TfidfTransformer()
        tfidf = tranform.fit_transform(vector.toarray())

        return  tfidf.toarray()