import math
import re
from typing import Sequence

Sentence = str
Corpus = list[Sentence]
Word = str
WordIdx = int
Vocab = dict[Word, WordIdx]
WordCount = (
    int | float
)  # this is awkward but necessary to not violate LSP in inheritance
DocTermMatrix = list[list[WordCount]]


class CountVectorizer:
    """Convert a collection of text documents to a matrix of token counts

    Attributes:
        doc_term_matrix (list): Matrix with frequency of terms in sentence
        vocab (dict): Mapping word to column index in document term matrix
    """

    def __init__(self):
        self.doc_term_matrix: DocTermMatrix = []
        self.vocab: Vocab = {}

    def fit_transform(self, corpus: Corpus) -> DocTermMatrix:
        """Fit on corpus and return document-term matrix"""
        self.doc_term_matrix, self.vocab = self._build_doc_term_matrix(corpus)
        return self.doc_term_matrix

    def _build_doc_term_matrix(
        self,
        corpus: Corpus,
    ) -> tuple[DocTermMatrix, Vocab]:
        """Building document-term matrix and corresponding vocabulary

        We build doc-term matrix while reading corpus in one pass,
        but this leaves matrix rows to have different size,
        so later we zero-pad this matrix to have length = vocabulary size

        Also we can replace vocab_size with len(word2dict), but this might
        be less clear than now
        """
        doc_term_matrix: DocTermMatrix = []
        word2idx: Vocab = {}
        vocab_size = 0
        for sentence in corpus:
            doc_term_vec = [0.0] * vocab_size
            for word in re.findall(r"\w{2,}", sentence.lower()):
                try:
                    doc_term_vec[word2idx[word]] += 1
                except KeyError:  # new word in vocab
                    doc_term_vec.append(1)
                    word2idx[word] = vocab_size  # always incremental value
                    vocab_size += 1

                assert len(doc_term_vec) == len(word2idx) == vocab_size, (
                    f"desync between {len(doc_term_vec)=}, "
                    f"{len(word2idx)=} and {vocab_size=}"
                )
            doc_term_matrix.append(doc_term_vec)

        # creates a new link to object doc_term_matrix, which is changed in place
        padded_doc_term_matrix = self._pad_doc_term_matrix(doc_term_matrix, vocab_size)

        return padded_doc_term_matrix, word2idx

    @staticmethod
    def _pad_doc_term_matrix(
        doc_term_matrix: DocTermMatrix,
        vocab_size: int,
    ) -> DocTermMatrix:
        """Right pad vectors in matrix with zero up to vocab_size"""
        for vec in doc_term_matrix:
            if len(vec) == vocab_size:
                break
            vec.extend([0] * (vocab_size - len(vec)))

        assert set(map(len, doc_term_matrix)) == {
            vocab_size
        }, "some vectors are not correctly padded"

        return doc_term_matrix

    def get_feature_names(self) -> list[Word]:
        """Get feature names from built vocabulary, ordered"""
        if not self.vocab:
            raise RuntimeError("Run fit_transform first")
        return list(self.vocab)

    def get_feature_names_out(self) -> list[Word]:
        """For compatibility with sklearn"""
        return self.get_feature_names()


TF = float
TFMatrix = list[list[TF]]
IDF = float
IDFVec = list[IDF]
TFIDFMatrix = list[list[float]]


def tf_transform(count_matrix: DocTermMatrix) -> TFMatrix:
    """Calculate term-frequency matrix from doc-term matrix"""
    tf_matrix = []
    for document in count_matrix:
        all_terms = sum(document)
        tf_vec = [cnt_term / all_terms for cnt_term in document]
        tf_matrix.append(tf_vec)
    return tf_matrix


def idf(count_matrix: DocTermMatrix) -> IDFVec:
    """Calculate inverse document frequency vector from doc-term matrix"""
    n_docs = len(count_matrix)

    def idf_formula(n_docs, n_docs_w_word) -> IDF:
        return math.log((n_docs + 1) / (n_docs_w_word + 1)) + 1

    def vec_tf_idf(
        cnt_word_in_all_docs: Sequence[int],
        n_docs: int = n_docs,
    ) -> IDF:
        n_docs_w_word = sum(1 for x in cnt_word_in_all_docs if x > 0)
        return idf_formula(n_docs, n_docs_w_word)

    idf_vec = list(map(vec_tf_idf, zip(*count_matrix)))
    return idf_vec


class TfIdfTransformer:
    """Transformer TF-IDF matrix from doc-term matrix"""

    def __init__(self) -> None:
        self.tf_idf_matrix: TFIDFMatrix = []

    def fit_transform(self, count_matrix: DocTermMatrix) -> TFIDFMatrix:
        """Build TF-IDF matrix from doc-term matrix"""
        tf_matrix = tf_transform(count_matrix)
        idf_vec = idf(count_matrix)

        for idx_doc in range(len(tf_matrix)):
            tf_matrix[idx_doc] = [  # changes lists in place
                tf * idf for tf, idf in zip(tf_matrix[idx_doc], idf_vec)
            ]

        self.tf_idf_matrix = tf_matrix
        return self.tf_idf_matrix


class TfidfVectorizer(CountVectorizer):
    """Vectorizer corpus of texts to TF-IDF matrix"""

    def __init__(self, transformer: TfIdfTransformer) -> None:
        self.transformer = transformer
        super().__init__()

    def fit_transform(self, corpus: Corpus) -> TFIDFMatrix:
        """Build TF-IDF matrix from corpus of texts"""
        count_matrix = super().fit_transform(corpus)
        tf_idf_matrix = self.transformer.fit_transform(count_matrix)
        return tf_idf_matrix


def round_tf_idf_matrix(tf_idf_matrix: TFIDFMatrix, round_digit=3):
    return [[round(tf_idf, round_digit) for tf_idf in doc] for doc in tf_idf_matrix]


if __name__ == "__main__":
    corpus = [
        "Crock Pot Pasta Never boil pasta again",
        "Pasta Pomodoro Fresh ingredients Parmesan to taste",
    ]
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())
    print(count_matrix)

    transformer = TfIdfTransformer()
    tfidf_vectorizer = TfidfVectorizer(transformer)
    tf_idf_matrix = tfidf_vectorizer.fit_transform(corpus)
    print(tf_idf_matrix)

    assert round_tf_idf_matrix(tf_idf_matrix) == [
        [0.201, 0.201, 0.286, 0.201, 0.201, 0.201, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.143, 0.0, 0.0, 0.0, 0.201, 0.201, 0.201, 0.201, 0.201, 0.201],
    ], "TF-IDF matrix is wrong"
