import re

Sentence = str
Corpus = list[Sentence]
Word = str
WordIdx = int
Vocab = dict[Word, WordIdx]
WordCount = int
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
        doc_term_matrix = []
        word2idx = dict()
        vocab_size = 0
        for sentence in corpus:
            doc_term_vec = [0] * len(word2idx)
            for word in re.findall(r"\w{2,}", sentence.lower()):
                try:
                    doc_term_vec[word2idx[word]] += 1
                except KeyError:
                    doc_term_vec.append(1)
                    word2idx[word] = vocab_size  # always incremental value
                    vocab_size += 1

                assert len(doc_term_vec) == len(word2idx) == vocab_size, (
                    f"desync between {len(doc_term_vec)=}, "
                    f"{len(word2idx)=} and {vocab_size=}"
                )

            doc_term_matrix.append(doc_term_vec)

        padded_doc_term_matrix = self._pad_doc_term_matrix(doc_term_matrix, vocab_size)

        return padded_doc_term_matrix, word2idx

    @staticmethod
    def _pad_doc_term_matrix(
        doc_term_matrix: DocTermMatrix,
        vocab_size: int,
    ) -> DocTermMatrix:
        """Right pad vectors in matrix with zero up to vocab_size"""
        for sentence in doc_term_matrix:
            if len(sentence) == vocab_size:
                break
            sentence.extend([0] * (vocab_size - len(sentence)))

        assert set(map(len, doc_term_matrix)) == {
            vocab_size
        }, "some vectors are not correctly padded"

        return doc_term_matrix

    def get_feature_names(self) -> list[str]:
        """Get feature names from built vocabulary, ordered"""
        if not self.vocab:
            raise RuntimeError("Run fit_transform first")
        return list(self.vocab)

    def toarray(self):
        """For compatibility with sklearn"""
        return self

    def get_feature_names_out(self):
        """For compatibility with sklearn"""
        return self.get_feature_names()


if __name__ == "__main__":
    corpus = [
        "Crock Pot Pasta Never boil pasta again",
        "Pasta Pomodoro Fresh ingredients Parmesan to taste",
    ]
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())
    print(count_matrix)
