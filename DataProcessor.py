from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class DataProcessor () :

    def __init__(self) :

        self.__vectorizer = CountVectorizer()
        self.__flag_fit = True


    def fit(self, corpus) :
        if not self.__flag_fit :
            raise RuntimeError("Only fit one corpus")
        vector = self.__vectorizer.fit_transform(corpus)
        tranform = TfidfTransformer()
        tfidf = tranform.fit_transform(vector.toarray())

        return  tfidf.toarray()

    def transform (self, corpus) :
        vocab = self.__vectorizer.get_feature_names()
        for doc in corpus:
            newdoc = []
            for word in doc:
                if word in vocab:
                    newdoc.append(word)
            doc = ' '.join(newdoc)
        vector = self.__vectorizer.transform(corpus)
        tranform = TfidfTransformer()
        tfidf = tranform.fit_transform(vector.toarray())

        return  tfidf.toarray()