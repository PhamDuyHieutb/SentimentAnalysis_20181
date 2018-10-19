from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
corpus = ['This is the first document.',
   'This document is the second document.',
  'And this is the third one.',
  'Is this the first document?'
    ]


# 'and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this'
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X)
transformer  = TfidfTransformer(smooth_idf= False)
tfidf = transformer.fit_transform(X)

print(tfidf.toarray())