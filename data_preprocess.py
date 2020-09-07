import csv

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
        with open(output_path, 'w') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow(('relation', 'pair'))
            pairs = 0
            for k, v in dict_relation.items():
                if len(v) > threshold:
                    pairs += 1
                    for w in v:
                        writer.writerow((k, w[0], w[1]))
            print('{} relations written to file'.format(pairs))


def main():
    path = "data/DErivBase-v2.0-rulePaths.txt"
    out = "data/out_threshold8.csv"
    preprocesser = PreprocesserDB(path)
    dr = preprocesser.read_file(star=True)
    preprocesser.write_to_file(out, dr)


if __name__ == "__main__":
    main()
