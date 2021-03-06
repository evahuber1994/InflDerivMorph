import numpy as np
import random
import torch
import torch.nn.functional as F
import csv
def make_vocabulary_matrix(feature_extractor, embedding_dim, target_words, restricted):
    if not restricted:
        vocabulary = feature_extractor.vocab.words
    else:
        whole_vocabulary = feature_extractor.vocab.words
        random.shuffle(whole_vocabulary)
        print("length voc: {}".format(len(whole_vocabulary)))
        length = int(len(whole_vocabulary)/10)
        print("length now: {}".format(str(length)))
        vocabulary = whole_vocabulary[:length]
        for t in target_words:
            if t not in vocabulary:
                vocabulary.append(t)
    lab2idx = {word: i + 1 for i, word in enumerate(vocabulary)}
    lab2idx['UNK'] = 0
    # dict(zip(self.vocabulary, range(len(self.vocabulary)))) #dict with vocabulary and indices
    idx2lab = {i: word for word, i in lab2idx.items()}
    vocabulary_matrix = np.zeros((len(vocabulary) + 1, embedding_dim))  # matrix with whole vocabulary
    vocabulary_matrix[0] = np.random.rand(1, embedding_dim)

    for i in range(len(vocabulary)):
        vocabulary_matrix[i + 1] = feature_extractor.get_embedding(vocabulary[i])

    return vocabulary_matrix, lab2idx, idx2lab

def shuffle_lists(zipped_lists):
    temp = list(zipped_lists)
    random.shuffle(temp)
    return zip(*temp)


def cosine_distance_loss(target_phrase, computed_phrase, dim=1, normalize=False):
    """
    Computes the cosine distance between two given phrases.
    """

    assert target_phrase.shape == computed_phrase.shape, "shapes of original and composed phrase have to be the same"
    if normalize:
        target_phrase = F.normalize(target_phrase, p=2, dim=dim)
        computed_phrase = F.normalize(computed_phrase, p=2, dim=dim)
    cosine_distances = 1 - F.cosine_similarity(target_phrase, computed_phrase, dim)
    total = torch.sum(cosine_distances)
    return total / target_phrase.shape[0]

def create_dict(path):
    dict_rel = dict()
    with open(path, 'r') as rf:
        next(rf)
        for l in rf:
            l = l.strip()
            if not l: continue
            line = l.split("\t")
            if line[0] in dict_rel:
                dict_rel[line[0]].append((line[1], line[2]))
            else:
                dict_rel[line[0]] = [(line[1], line[2])]
    return dict_rel

def save_new(dict_rel, out_path):
    with open(out_path, 'w') as wf:
        writer = csv.writer(wf, delimiter="\t")
        writer.writerow(("relation", "w1", "w2"))
        for k,v in dict_rel.items():
            for w in v:
                writer.writerow((k, w[0], w[1]))