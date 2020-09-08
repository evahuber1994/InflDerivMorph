import spacy
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import torch
import finalfusion
import numpy as np

def read_deriv(path):
    relations = []
    word1 = []
    word2 = []
    with open(path) as file:
        next(file)
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

    def __init__(self, feature_extractor, word1, word2):  # batch_size?

        #self._nlp = spacy.load(embedding)
        self._feature_extractor = feature_extractor
        self._word1 = word1  # list of words
        self._word2 = word2 #list of derived words
        self._samples = self.produce_samples()

    #def get_static(self, word):
    #    return torch.from_numpy(self.nlp(word).vector)

    def produce_samples(self):
        return [{'w1': self.feature_extractor.get_embedding(x), 'w2':self.feature_extractor.get_embedding(y), 'w1_form': x, 'w2_form': y} for x, y in zip(self.word1, self.word2)]
        #return [{'w1': self.get_static(x), 'w2': self.get_static(x)} for x, y in zip(self.word1, self.word2)]

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
    def feature_extractor(self):
        return self._feature_extractor


class FeatureExtractor:
    def __init__(self, path_to_embeddings):
        if path_to_embeddings.endswith("fifu"):
            self._embeds = finalfusion.load_finalfusion(path_to_embeddings)
        elif path_to_embeddings.endswith("bin"):
            self._embeds = finalfusion.load_fasttext(path_to_embeddings)
        elif path_to_embeddings.endswith("w2v"):
            self._embeds = finalfusion.load_word2vec(path_to_embeddings)
        else:
            print("attempt to read invalid embeddings")

    def get_embedding(self, word):
        """
        takes a word and returns its embedding
        :param word: the word for which an embedding should be returned
        :return: the embedding of the word or random embedding if word not in vocab
        """
        embedding = self._embeds.embedding(word)
        if embedding is None:
            print("found unknown word : ", word)
            embedding = np.random.rand(self.embedding_dim).astype(np.float32)
        return embedding

    def get_array_embeddings(self, array_words):
        """
        takes an array of words and returns an array of embeddings of those words
        :param array_words: a word array of length x
        :return: array_embeddings: the embeddings of those words in an array of length x
        """
        array_embeddings = []
        [array_embeddings.append(self.get_embedding(words)) for words in array_words]
        return array_embeddings

    @property
    def embeds(self):
        return self._embeds

    @property
    def embedding_dim(self):
        return self._embedding_dim

def main():
    emb = 'de_core_news_sm'
    path = 'data/out_threshold8.csv'
    _, words, labels = read_deriv(path)

    extractor = SimpleDataLoader(emb, words[1:10], labels[1:10])
    for i,j in extractor:
        print(i)



if __name__ == "__main__":
    main()
