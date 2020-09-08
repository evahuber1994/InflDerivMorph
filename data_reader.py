import spacy
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import torch

def read_deriv(path):
    relations = []
    word1 = []
    word2 = []
    with open(path) as file:
        for l in file:
            l = l.strip()
            if not l: continue
            line = l.split('\t')
            relations.append(line[0])
            word1.append(line[1])
            word2.append(line[2])
    return relations, word1, word2

#not needed at the moment, there are no labels
def create_label_encoder(all_labels):
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    return label_encoder


class SimpleDataLoader(Dataset):

    def __init__(self, embedding, word1, word2):  # batch_size?

        self._nlp = spacy.load(embedding)
        self._word1 = word1  # list of words
        self._word2 = word2 #list of derived words
        self._samples = self.produce_samples()

    def get_static(self, word):
        return torch.from_numpy(self.nlp(word).vector)

    def produce_samples(self):

        return [{'w1': self.get_static(x), 'w2': self.get_static(x)} for x, y in zip(self.word1, self.word2)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



    @property
    def word1(self):
        return self._word1

    @property
    def samples(self):
        return self._samples

    @property
    def word2(self):
        return self._word2

    @property
    def nlp(self):
        return self._nlp


def main():
    emb = 'de_core_news_sm'
    path = 'data/out_threshold8.csv'
    _, words, labels = read_deriv(path)

    extractor = SimpleDataLoader(emb, words[1:10], labels[1:10])
    for i,j in extractor:
        print(i)



if __name__ == "__main__":
    main()
