import csv
from data_reader import read_deriv, FeatureExtractor, GensimFeatureExtractor
import random
import os
import pandas as pd
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
                        w1 = tup[0].split('_')
                        w2 = tup[1].split('_')
                        pair = (w1[0], w2[0])
                        print(pair)
                        if pair in word_pairs:
                            continue
                        if element not in dict_relations:
                            dict_relations[element] = [pair]
                            word_pairs.add(pair)
                        else:
                            dict_relations[element].append(pair)
                            word_pairs.add(pair)
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
            writer.writerow(('relation', 'w1', 'w2'))
            pairs = 0
            for k, v in dict_relation.items():
                if self.oov:  # if oov is true, then skip all words that either base or derived form is not in vocabulary
                    i = 0
                    for w in v:
                        if w[0] not in self.embedding_voc or w[1] not in self.embedding_voc:
                            i +=1
                if len(v)-i > self.threshold:
                    for w in v:
                        w1 = w[0]
                        w2 = w[1]
                        # if (w1[0], w2[0]) not in word_pairs:
                        if self.oov:  # if oov is true, then skip all words that either base or derived form is not in vocabulary
                            if w1 not in self.embedding_voc or w2 not in self.embedding_voc:
                                continue
                        pairs += 1
                        writer.writerow((k, w1, w2))
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


def make_splits(data_path, out_path, size_train=0.6, size_test=0.2, wm='w'):
    write_method = wm
    relations, word1, word2 = read_deriv(data_path)
    data = [(r, w1, w2) for r, w1, w2 in zip(relations, word1, word2)]

    random.shuffle(data)

    train_length = int(len(data) * size_train)
    test_length = int(len(data) * size_test)
    train = data[:train_length]
    val = data[train_length:train_length + test_length]
    test = data[train_length + test_length:]
    with open(out_path + '_train.csv', write_method) as trf:
        writer = csv.writer(trf, delimiter='\t')
        if write_method =='w':
            writer.writerow(("relation", "word1", "word2"))
        for l in train:
            writer.writerow((l[0], l[1], l[2]))
    with open(out_path + '_val.csv',write_method) as vf:
        writer = csv.writer(vf, delimiter='\t')
        if write_method == 'w':
            writer.writerow(("relation", "word1", "word2"))
        for l in val:
            writer.writerow((l[0], l[1], l[2]))
    with open(out_path + '_test.csv', write_method) as tef:
        writer = csv.writer(tef, delimiter='\t')
        if write_method == 'w':
            writer.writerow(("relation", "word1", "word2"))
        for l in test:
            writer.writerow((l[0], l[1], l[2]))


def make_splits_in_one(directory, out_path,size_train, size_test):
    for fn in os.listdir(directory):
        path = os.path.join(directory, fn)
        make_splits(path, out_path, size_train, size_test, wm='a')

def make_splits_in_dir(directory, out_path, size_train, size_test):
    for fn in os.listdir(directory):
        path = os.path.join(directory, fn)
        new_path = os.path.join(out_path, fn.strip(".csv"))
        os.makedirs(new_path)

        out = os.path.join(new_path, fn.strip('.csv'))
        # out =  os.path.join(out_path, new_dir, str(fn.strip('csv')))
        make_splits(path, out, size_train, size_test, wm='w')

def shorten_files(path_in, path_out, length):
    df_in = pd.read_csv(path_in, delimiter='\t')
    df_in = df_in.sample(frac=1)[:length]

    df_in.to_csv(path_out, sep='\t', index=False)

def make_splits_balanced(path_in, path_out, train_size = 0.6, test_size= 0.2):
    dict_relations = dict()
    with open(path_in, 'r') as file:
        for l in file:
            l = l.strip()
            if not l: continue
            line = l.split('\t')
            if line[0] in dict_relations:
                dict_relations[line[0]].append((line[1], line[2]))
            else:
                dict_relations[line[0]] = [(line[1], line[2])]
    train_path = path_out + "train.csv"
    val_path = path_out + "val.csv"
    test_path = path_out+ "test.csv"
    header = ("relation", "w1", "w2")
    f_train = open(train_path, 'w')
    writer_train = csv.writer(f_train, delimiter='\t')
    writer_train.writerow(header)
    f_val = open(val_path, 'w')
    writer_val = csv.writer(f_val, delimiter="\t")
    writer_val.writerow(header)
    f_test = open(test_path, 'w')
    writer_test = csv.writer(f_test, delimiter="\t")
    writer_test.writerow(header)
    for k,v in dict_relations.items():
        train_length = int(len(v) * train_size)
        test_length = int(len(v) * test_size)
        train =v[:train_length]
        val = v[train_length:train_length +test_length]
        test = v[train_length + test_length:]
        for w in train:
            writer_train.writerow((k, w[0], w[1]))
        for w in val:
            writer_val.writerow((k, w[0], w[1]))
        for w in test:
            writer_test.writerow((k, w[0], w[1]))
    f_test.close()
    f_train.close()
    f_val.close()





def main():

    path_in = '/home/evahu/Documents/Master/Master_Dissertation/data_conll/INFLECTION/inf_conll_thr80.csv'
    path_out = '/home/evahu/Documents/Master/Master_Dissertation/data_conll/INFLECTION/splits/'

    make_splits_balanced(path_in, path_out)

if __name__ == "__main__":
    main()
