import csv
from data_reader import read_deriv, FeatureExtractor
import random
import os

"""
preprocessing file to extract all pairs that share a relation from DErivBase
boolean in read method to indicate whether inverse relation marked by star should be added or not
"""


class Preprocesser():
    def __init__(self, path, out_path, embedding_voc, star=True, OOV=True, threshold=80):
        self._path = path
        self._out_path = out_path
        self._star = star
        self._oov = OOV
        self._embedding_voc = embedding_voc
        self._threshold = threshold

    def read_and_write_DB(self):
        """
        :param star: if true, inverse relations indicated by star won't be added
        :return: dict_pairs: dictionary of related pairs, keys are relation (patterns)
        """
        word_pairs = set()
        dict_relations = dict()
        with open(self.path, 'r') as file:
            for l in file:
                l = l.strip()
                if not l: continue
                l = l.split()[3:]
                for i, element in enumerate(l):
                    true_cond = element.startswith('d') and element.endswith('>')
                    if self.star == True:
                        true_cond = element.startswith('d') and element.endswith('>') and '*' not in element
                    if true_cond:
                        element = element.strip('>')
                        element = element.split('.')[0]
                        tup = (l[i - 1], l[i + 1])
                        if tup in word_pairs:
                            continue
                        if element not in dict_relations:
                            dict_relations[element] = [tup]
                            word_pairs.add(tup)
                        else:
                            dict_relations[element].append(tup)
                            word_pairs.add(tup)
        self.write_to_fileDB(self.out_path, dict_relations)
        return dict_relations

    # helper method
    def write_to_fileDB(self, output_path, dict_relation):
        """
        :param output_path: path to write file to
        :param dict_relation: dictionary containing relations and word pairs
        :param threshold: threshold of nr of word pairs for one relation
        """
        # word_pairs = set()
        with open(output_path, 'w') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow(('relation', 'w1', 'w2', 'catw1', 'catw2'))
            pairs = 0
            for k, v in dict_relation.items():
                if len(v) > self.threshold:
                    for w in v:
                        w1 = w[0].split('_')
                        w2 = w[1].split('_')
                        # if (w1[0], w2[0]) not in word_pairs:
                        if self.oov:  # if oov is true, then skip all words that either base or derived form is not in vocabulary
                            if w1[0] not in self.embedding_voc or w2[0] not in self.embedding_voc:
                                continue
                        pairs += 1
                        writer.writerow((k, w1[0], w2[0], w1[1], w2[1]))
                        # word_pairs.add((w1[0], w2[0]))
                else:
                    continue
            print('{} relations written to file'.format(pairs))

    def read_and_write_unimorph(self):
        """
      method that transforms unimorph file into same format as PreprocesserDB
      """
        dict_relations = self.get_dict_unimorph()
        with open(self.out_path, 'w') as wf:
            writer = csv.writer(wf, delimiter='\t')
            writer.writerow(("relation", "word1", "word2"))
            for k, v in dict_relations.items():
                if len(v) > self.threshold:
                    print(k, len(v))
                    for w in v:
                        writer.writerow((k, w[0], w[1]))
        return dict_relations

    # helper method
    def get_dict_unimorph(self):
        dict_relations = dict()
        with open(self.path) as rf:
            for l in rf:
                l = l.strip()
                if not l: continue
                line = l.split('\t')
                if ' ' in line[0] or ' ' in line[1]:
                    print(line[1])
                    continue
                if self.oov:  # if oov is true, then skip all words that either base or derived form is not in vocabulary
                    if line[1] not in self.embedding_voc or line[2] not in self.embedding_voc:
                        continue
                if line[0] in dict_relations:
                    dict_relations[line[0]].append((line[1], line[2]))
                else:
                    dict_relations[line[0]] = [(line[1], line[2])]
        return dict_relations

    @property
    def out_path(self):
        return self._out_path

    @property
    def path(self):
        return self._path

    @property
    def star(self):
        return self._star

    @property
    def oov(self):
        return self._oov

    @property
    def embedding_voc(self):
        return self._embedding_voc

    @property
    def threshold(self):
        return self._threshold


"""
method used to create new files: one file per relation
files already have to be in the appropriate format 
"""


def make_relation_files(data_path, out_path):
    relations, word1, word2 = read_deriv(data_path)
    dict_rels = dict()
    for r, w1, w2 in zip(relations, word1, word2):
        if r not in dict_rels:
            dict_rels[r] = [(w1, w2)]
        else:
            dict_rels[r].append((w1, w2))
    for k, v in dict_rels.items():
        out = os.path.join(out_path, str(k) + ".csv")
        with open(out, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(('relation', 'w1', 'w2'))
            for w in v:
                writer.writerow((k, w[0], w[1]))


"""
method used to create splits in directories, one relation in a directory, three files in one directory (train, val, test)
"""


def make_splits(data_path, out_path, size_train=0.6, size_test=0.2):
    relations, word1, word2 = read_deriv(data_path)
    data = [(r, w1, w2) for r, w1, w2 in zip(relations, word1, word2)]

    random.shuffle(data)

    train_length = int(len(data) * size_train)
    test_length = int(len(data) * size_test)
    train = data[:train_length]
    val = data[train_length:train_length + test_length]
    test = data[train_length + test_length:]
    with open(out_path + '_train.csv', 'w') as trf:
        writer = csv.writer(trf, delimiter='\t')
        writer.writerow(("relation", "word1", "word2"))
        for l in train:
            writer.writerow((l[0], l[1], l[2]))
    with open(out_path + '_val.csv', 'w') as vf:
        writer = csv.writer(vf, delimiter='\t')
        writer.writerow(("relation", "word1", "word2"))
        for l in val:
            writer.writerow((l[0], l[1], l[2]))
    with open(out_path + '_test.csv', 'w') as tef:
        writer = csv.writer(tef, delimiter='\t')
        writer.writerow(("relation", "word1", "word2"))
        for l in test:
            writer.writerow((l[0], l[1], l[2]))


def make_splits_in_dir(directory, out_path, size_train, size_test):
    for fn in os.listdir(directory):
        path = os.path.join(directory, fn)
        new_path = os.path.join(out_path, fn.strip(".csv"))
        os.makedirs(new_path)

        out = os.path.join(new_path, fn.strip('.csv'))
        # out =  os.path.join(out_path, new_dir, str(fn.strip('csv')))
        make_splits(path, out, size_train, size_test)
    # dir = 'data/inflection/files_per_relation/not_split'
    # for fn in os.listdir(dir):
    #    path = os.path.join(dir, fn)
    #    out_path = 'data/inflection/files_per_relation/split/' + str(fn.strip('.csv'))
    #    make_splits(path, out_path)


def main():
    """
    path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data2/DERIVATION/DErivBase-v2.0-rulePaths.txt'
    out_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data2/DERIVATION/deriv_thresh80.csv'
    embeddings = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/embeddings/word2vec-mincount-30-dims-100-ctx-10-ns-5.w2v'
    voc = FeatureExtractor(embeddings, 200)

    # path,out_path, embedding_voc, star=True, OOV =True, threshold=80
    prep = Preprocesser(path, out_path, voc.vocab, star=True, OOV=True, threshold=80)
    prep.read_and_write_DB()


    path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data2/DERIVATION/deriv_thresh80.csv'
    out_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data2/DERIVATION/relation_files'
    make_relation_files(path,out_path)
    """
    """
    data_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data2/DERIVATION/relation_files'
    out_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data2/DERIVATION/split_relations'
    make_splits_in_dir(data_path, out_path, size_train = 0.6, size_test = 0.2)
    """
    #####UNIMORPH
    """
    path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data2/INFLECTION/deu_formatted.csv'
    out_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data2/INFLECTION/inf_formatted80.csv'
    embeddings = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/embeddings/word2vec-mincount-30-dims-100-ctx-10-ns-5.w2v'
    voc = FeatureExtractor(embeddings, 200)
    prep = Preprocesser(path, out_path, voc.vocab, star=True, OOV=True, threshold=80)
    dict_r = prep.read_and_write_unimorph()
    print(dict_r['V;SBJV;PST;2;PL'])
    """
    #path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data2/INFLECTION/inf_formatted80.csv'
    #out_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data2/INFLECTION/relations'
    #make_relation_files(path,out_path)

    data_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data2/INFLECTION/relations'
    out_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data2/INFLECTION/split_relations'
    make_splits_in_dir(data_path, out_path, size_train = 0.6, size_test = 0.2)


if __name__ == "__main__":
    main()
