from utils import create_dict
import os
import argparse


def counts(path):
    total_nr = 0
    nr_per_rel = []
    dict_rel = create_dict(path)
    nr_of_rels = len(dict_rel)
    for k, v in dict_rel.items():
        nr_per_rel.append((k, len(v)))
        total_nr += len(v)
    return total_nr, nr_of_rels, nr_per_rel


def save_info(dict_info, out_path, keyword):
    with open(out_path, 'w') as wf:
        wf.write("metadata for: {}\n".format(keyword))
        wf.write("{}\t{}\t{}\n".format("split", "nr of data points", "nr of relations"))
        for k, v in dict_info.items():
            tot_nr = v[0]
            nr_rels = v[1]
            wf.write("{}\t{}\t{}\n".format(k, str(tot_nr), str(nr_rels)))


def save_relations(dict_info, out_path, keyword):
    with open(out_path, 'w') as wf:
        wf.write("all relations for: {}\n \n \n".format(keyword))
        wf.write("{}\t{}\t{}\n".format("split", "relation", "nr of data points"))
        for k, v in dict_info.items():
            rels = v[2]
            for r in rels:
                wf.write("{}\t{}\t{}\n".format(k, r[0], r[1]))


def main(path, keyword):
    out_path = os.path.join(path, "README.txt")
    out_path_rels = os.path.join(path, "relations.txt")
    dict_info = dict()
    for f in os.listdir(path):
        print(f)
        name = f.strip(".csv").strip("combined_")
        print(name)
        path_file = os.path.join(path, f)
        tot_nr, nr_rels, nr_membs = counts(path_file)
        dict_info[name] = (tot_nr, nr_rels, nr_membs)

    save_info(dict_info, out_path, keyword)
    save_relations(dict_info, out_path_rels, keyword)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_directory")
    parser.add_argument("dataset_name")
    args = parser.parse_args()
    main(args.path_directory, args.dataset_name)
