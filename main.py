from collections import defaultdict


class CountVectorizer:
    def __init__(self):
        self.feature_names = []

    def fit_transform(self, corpus):
        cnt, vocab = self._word_cnt(corpus)
        self.vocab = vocab
        return cnt

    @staticmethod
    def _word_cnt(corpus):
        word_cnt_corpus = []
        # vocab = set()
        vocab = []
        for sentence in corpus:
            word_cnt_dict = defaultdict(int)
            for word in sentence.split():
                word_cnt_dict[word.lower()] += 1
                if word.lower() not in vocab:
                    vocab.append(word.lower())
            word_cnt_corpus.append(dict(word_cnt_dict))

            # vocab.update(set(word_cnt_dict.keys()))

        vocab = dict(zip(vocab, range(len(vocab))))

        size = len(vocab)
        id_cnt = []
        for sentence in word_cnt_corpus:
            ll = [0] * size
            for word in sentence:
                ll[vocab[word]] += sentence[word]
            id_cnt.append(ll)

        return id_cnt, vocab

    def get_feature_names(self):
        return self.vocab
        return list(self.vocab)


if __name__ == "__main__":
    corpus = [
        "Crock Pot Pasta Never boil pasta again",
        "Pasta Pomodoro Fresh ingredients Parmesan to taste",
    ]
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())
    print(count_matrix)
