import re
from collections import defaultdict


class CountVectorizer:
    """Convert a collection of text documents to a matrix of token counts"""

    def __init__(self):
        self.feature_names = []
        self.vocab = {}

    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        doc_term_matrix, vocab = self._word_cnt2(corpus)
        self.vocab = vocab
        return doc_term_matrix

    @staticmethod
    def _word_cnt(corpus):
        word_cnt_corpus = []
        vocab = set()
        for sentence in corpus:
            word_cnt_dict = defaultdict(int)
            for word in re.findall(r"\w{2,}", sentence.lower()):
                word_cnt_dict[word] += 1
            word_cnt_corpus.append(dict(word_cnt_dict))

            vocab.update(set(word_cnt_dict.keys()))

        vocab = dict(zip(vocab, range(len(vocab))))

        size = len(vocab)
        doc_term_matrix = []
        for sentence in word_cnt_corpus:
            ll = [0] * size
            for word in sentence:
                ll[vocab[word]] += sentence[word]
            doc_term_matrix.append(ll)

        return doc_term_matrix, vocab

    @staticmethod
    def _word_cnt2(corpus):
        word_cnt_corpus = []
        vocab = dict()
        vocab_size = 0
        for sentence in corpus:
            doc_term_vec = [0] * len(vocab)
            word_cnt_dict = defaultdict(int)
            for word in re.findall(r"\w{2,}", sentence.lower()):
                if word in vocab:
                    doc_term_vec[vocab[word]] += 1
                else:
                    doc_term_vec.append(1)
                    vocab[word] = vocab_size
                    vocab_size += 1
                word_cnt_dict[word] += 1
            word_cnt_corpus.append(doc_term_vec)
        for num_sentence in range(len(word_cnt_corpus)):
            if len(word_cnt_corpus[num_sentence]) == vocab_size:
                break
            cur_vec = word_cnt_corpus[num_sentence]
            cur_len = len(cur_vec)
            cur_vec.extend([0] * (vocab_size - cur_len))

        return word_cnt_corpus, vocab

    def get_feature_names(self) -> list[str]:
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
