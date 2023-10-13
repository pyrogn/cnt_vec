from sklearn.feature_extraction.text import CountVectorizer
from main import CountVectorizer as CustomCountVectorizer
import numpy as np
import time

corpus = [
    "This is the first document.",
    "This document is the second document. mr. bobby's hour-second ",
    "And this is the third one. a cat ",
    "Is this the first document?",
]
with open("data/frankenstein.txt", "r") as f:
    corpus = f.read().split(".")
st1 = time.perf_counter()
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
vocab = list(vectorizer.get_feature_names_out())
en1 = time.perf_counter()
# print(X.toarray())
# print(vectorizer.get_feature_names_out())
# print(X.toarray().shape, type(X.toarray()), type(X.toarray()[0]))
# print(vectorizer.get_feature_names_out().shape)
st2 = time.perf_counter()
vectorizer_custom = CustomCountVectorizer()
X_custom = vectorizer_custom.fit_transform(corpus)
# print(np.array(X_custom).shape)
vocab_custom = vectorizer_custom.get_feature_names_out()
# print(len(vectorizer_custom.get_feature_names()))
en2 = time.perf_counter()
t1 = en1 - st1
t2 = en2 - st2
print(t1, t2, t2 / t1)
# vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
# X2 = vectorizer2.fit_transform(corpus)
# vectorizer2.get_feature_names_out()

assert len(vectorizer.get_feature_names_out()) == len(
    vectorizer_custom.get_feature_names_out()
), "different len"
assert not (
    set(vectorizer.get_feature_names_out())
    ^ set(vectorizer_custom.get_feature_names_out())
), "some extra words: "

X_custom = np.array(X_custom)[:, [vocab_custom.index(word) for word in vocab]]
assert np.sum(X.toarray() != X_custom) == 0, "matrices are not equal"
