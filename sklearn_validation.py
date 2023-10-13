"""Measure speed comparison with CountVectorizer form sklearn
Corpus - Frankestein (441k characters, 7.6k lines, 3.1k sentences"""
from sklearn.feature_extraction.text import CountVectorizer
from main import CountVectorizer as CustomCountVectorizer
import numpy as np
import time
import subprocess

_simple_corpus = [
    "This is the first document.",
    "This document is the second document. mr. bobby's second-hand ",
    "And this is the third one. a cat ",
    "Is this the first document?",
]

with open("data/frankenstein.txt", "r") as f:
    corpus = f.read().split(".")

vectorizer = CountVectorizer()

st1 = time.perf_counter()
X = vectorizer.fit_transform(corpus)
vocab = vectorizer.get_feature_names_out()
en1 = time.perf_counter()

vocab = list(vocab)

vectorizer_custom = CustomCountVectorizer()

st2 = time.perf_counter()
X_custom = vectorizer_custom.fit_transform(corpus)
vocab_custom = vectorizer_custom.get_feature_names_out()
en2 = time.perf_counter()

t1 = en1 - st1
t2 = en2 - st2


assert len(vocab) == len(
    vocab_custom
), f"different len in vocabulary, {len(vocab)} sklearn vs {len(vocab_custom)} custom"

assert not (
    set(vocab) ^ set(vocab_custom)
), f"some extra words: {set(vocab) ^ set(vocab_custom)}"

X_np = X.toarray()
vocab_custom_dict = dict(zip(vocab_custom, range(len(vocab_custom))))
X_custom_np = np.array(X_custom)[:, [vocab_custom_dict[word] for word in vocab]]
assert (
    np.sum(X_np != X_custom_np) == 0
), "doc-term matrices are not equal, sorry, you're on your own"


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


with open("stats.csv", "r") as f:
    last_line = f.read().strip().split("\n")[-1]
    last_git_hash_stats = last_line.split(";")[0]
    cur_git_hash = get_git_revision_short_hash()

    if last_git_hash_stats != cur_git_hash:
        with open("stats.csv", "a") as f:
            values = [
                cur_git_hash,
                round(t1, 4),
                round(t2, 4),
                round(t2 / t1, 2),
            ]
            values_str = map(str, values)
            print(";".join(values_str), file=f)
