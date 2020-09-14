import numpy as np


def make_vocabulary_matrix(feature_extractor, embedding_dim):
    vocabulary = feature_extractor.vocab.words
    lab2idx = {word: i + 1 for i, word in enumerate(vocabulary)}
    lab2idx['UNK'] = 0
    # dict(zip(self.vocabulary, range(len(self.vocabulary)))) #dict with vocabulary and indices
    idx2lab = {i: word for word, i in lab2idx.items()}
    vocabulary_matrix = np.zeros((len(vocabulary) + 1, embedding_dim))  # matrix with whole vocabulary
    vocabulary_matrix[0] = np.random.rand(1, embedding_dim)

    for i in range(len(vocabulary)):
        vocabulary_matrix[i + 1] = feature_extractor.get_embedding(vocabulary[i])

    return vocabulary_matrix, lab2idx, idx2lab