import numpy as np
import pandas as pd

from keras.preprocessing import text, sequence


class ToxicComments:
    def __init__(self, train_file, test_file):
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)

        train["comment_text"].fillna("fillna")
        test["comment_text"].fillna("fillna")

        self.X_train = train["comment_text"].str.lower()
        self.X_test = test["comment_text"].str.lower()

        self.y_train = train[
            ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        ].values

    def tokenize(self, max_features, max_len, embed_size):
        tokenizer = text.Tokenizer(num_words=max_features, lower=True)
        tokenizer.fit_on_texts(list(self.X_train) + list(self.X_test))

        X_train = tokenizer.texts_to_sequences(self.X_train)
        X_test = tokenizer.texts_to_sequences(self.X_test)

        self.X_train = sequence.pad_sequences(X_train, maxlen=max_len)
        self.X_test = sequence.pad_sequences(X_test, maxlen=max_len)

        return tokenizer

    def make_embedding_matrix(
        self, tokenizer, embedding_file, max_features, embed_size
    ):

        embeddings_index = {}
        with open(embedding_file, encoding="utf8") as f:
            for line in f:
                values = line.rstrip().rsplit(" ")
                word = values[0]
                coefs = np.asarray(values[1:], dtype="float32")
                embeddings_index[word] = coefs

        word_index = tokenizer.word_index

        num_words = min(max_features, len(word_index) + 1)
        embedding_matrix = np.zeros((num_words, embed_size))

        for word, i in word_index.items():
            if i >= max_features:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix
