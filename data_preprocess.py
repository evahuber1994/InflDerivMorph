import csv
from data_reader import read_deriv
import random
import os

"""
preprocessing file to extract all pairs that share a relation from DErivBase
boolean in read method to indicate whether inverse relation marked by star should be added or not
"""


class PreprocesserDB():
    def __init__(self, path):
        self.path = path

    def read_file(self, star=True):
        """
        :param star: if true, inverse relations indicated by star won't be added
        :return: dict_pairs: dictionary of related pairs, keys are relation (patterns)
        """
        dict_relations = dict()
        with open(self.path, 'r') as file:
            for l in file:
                l = l.strip()
                if not l: continue
                l = l.split()[3:]
                for i, element in enumerate(l):
                    true_cond = element.startswith('d') and element.endswith('>')
                    if star == True:
                        true_cond = element.startswith('d') and element.endswith('>') and '*' not in element
                    if true_cond:
                        element = element.strip('>')
                        element = element.split('.')[0]
                        tup = (l[i - 1], l[i + 1])
                        if element not in dict_relations:
                            dict_relations[element] = [tup]
                        else:
                            dict_relations[element].append(tup)
        return dict_relations

    def write_to_file(self, output_path, dict_relation, threshold=80):
        """
        :param output_path: path to write file to
        :param dict_relation: dictionary containing relations and word pairs
        :param threshold: threshold of nr of word pairs for one relation
        """
        word_pairs = set()
        with open(output_path, 'w') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow(('relation', 'w1', 'w2', 'catw1', 'catw2'))
            pairs = 0
            for k, v in dict_relation.items():
                if len(v) > threshold:
                    pairs += 1
                    for w in v:
                        w1 = w[0].split('_')
                        w2 = w[1].split('_')
                        if (w1[0], w2[0]) not in word_pairs:
                            writer.writerow((k, w1[0], w2[0], w1[1], w2[1]))
                            word_pairs.add((w1[0], w2[0]))
            print('{} relations written to file'.format(pairs))

"""
method used to create new files: one file per relation
"""
def make_relation_files(data_path, out_path):
    relations, word1, word2 = read_deriv(data_path)
    dict_rels = dict()
    for r, w1, w2 in zip(relations, word1, word2):
        if r not in dict_rels:
            dict_rels[r] = [(w1,w2)]
        else:
            dict_rels[r].append((w1,w2))
    for k, v in dict_rels.items():
        with open(out_path + str(k) + '.csv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(('relation', 'w1', 'w2'))
            for w in v:
                writer.writerow((k, w[0], w[1]))

def make_splits(data_path, out_path):
    relations, word1, word2 = read_deriv(data_path)
    data = [(r, w1, w2) for r, w1, w2 in zip(relations, word1, word2)]

    random.shuffle(data)

    train_length = int(len(data) * 0.6)
    test_length = int(len(data) * 0.2)
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


def main():
    """
    to preprocess raw files
    """
    # path = "data/DErivBase-v2.0-rulePaths.txt"
    # out = "data/out_threshold8.csv"
    # preprocesser = PreprocesserDB(path)
    # dr = preprocesser.read_file(star=True)
    # preprocesser.write_to_file(out, dr)

    """
    to make splits from preprocessed files 
    """
    #path = 'data/out_threshold8.csv'
    #out = 'data/splits/out_threshold8'
    #make_splits(path, out)

    """
    to create relation individual files
    """
    #path = 'data/out_threshold8.csv'
    #out = 'data/files_per_relation/not_split/'
    #make_relation_files(path, out)

    """
    to make splits from relation individual files
    """
    #dir = 'data/files_per_relation/not_split'
    #for fn in os.listdir(dir):
    #    path = os.path.join(dir, fn)
    #    out_path = 'data/files_per_relation/splits/' + str(fn.strip('.csv'))
    #    make_splits(path, out_path)



if __name__ == "__main__":
    main()
