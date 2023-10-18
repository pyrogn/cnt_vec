"""Measure speed comparison with CountVectorizer form sklearn

Corpus - Frankenstein (441k characters, 7.6k lines, 3.1k sentences)

This looks quite a mess, but I think it's ok for now
"""
import functools
import subprocess
import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore

from main import CountVectorizer as CustomCountVectorizer

_simple_corpus = [
    "This is the first document.",
    "This document is the second document. mr. bobby's second-hand ",
    "And this is the third one. a cat ",
    "Is this the first document?",
]

with open("data/frankenstein.txt", "r") as f:
    corpus = f.read().split(".")


def log_time(fn):
    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = fn(*args, **kwargs)
        end_time = time.perf_counter()
        time_total = end_time - start_time
        return result, time_total

    return _wrapper


@log_time
def vectorizer_run_log(vectorizer, corpus):
    X = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names_out()
    return X, features


def unify_X_features(X, features, *, is_sklearn: bool):
    if is_sklearn:
        X = X.toarray()
        features = list(features)
    X = np.array(X)
    return X, features


vectorizer = CountVectorizer()
(X, features), t1 = vectorizer_run_log(vectorizer, corpus)
X, features = unify_X_features(X, features, is_sklearn=True)

vectorizer_custom = CustomCountVectorizer()
(X_custom, features_custom), t2 = vectorizer_run_log(vectorizer_custom, corpus)
X_custom, features_custom = unify_X_features(
    X_custom, features_custom, is_sklearn=False
)


assert len(features) == len(features_custom), (
    f"different len in vocabulary:"
    f" {len(features)} sklearn vs {len(features_custom)} custom"
)

xor_words = set(features) ^ set(features_custom)
assert not xor_words, f"some extra words: {xor_words}"

vocab_custom_dict = dict(zip(features_custom, range(len(features_custom))))
X_custom_same_order = X_custom[:, [vocab_custom_dict[word] for word in features]]

assert X.shape == X_custom_same_order.shape, (
    f"doc-term matrices shapes are not equal: "
    f"{X.shape} sklearn vs {X_custom_same_order.shape} custom"
)

assert (
    np.sum(X != X_custom_same_order) == 0
), "doc-term matrices are not equal, sorry, you're on your own"


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


with open("stats.csv", "r") as fr:
    last_line = fr.read().strip().split("\n")[-1]
    last_git_hash_stats = last_line.split(";")[0]
    cur_git_hash = get_git_revision_short_hash()

    if last_git_hash_stats != cur_git_hash:
        with open("stats.csv", "a") as fa:
            values = [
                cur_git_hash,
                round(t1, 4),
                round(t2, 4),
                round(t2 / t1, 2),
            ]
            values_str = map(str, values)
            print(";".join(values_str), file=fa)
