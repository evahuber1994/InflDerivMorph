from utils import create_dict
import os
import argparse
import pandas as pd
def average_counts(path):
    df = pd.read_csv(path, sep="\t")
    print(df.keys())
    infs = df[df['relation'].str.startswith("INF_")]
    ders = df[df['relation'].str.startswith("DER_")]
    out_path = path.replace(".csv", "_average.csv")
    #out_path_der = path.replace(".csv", "average_DER.csv")
    infs_means = infs.sum()
    infs_means = infs_means[1:]
    infs_means.loc[-1] = ['set_infs', 'number']
    ders_means = ders.sum()
    ders_means = ders_means[1:]
    ders_means.loc[-1] = ['set_ders', 'number']
    both_means = pd.concat([infs_means, ders_means])
    print(both_means)
    both_means.to_csv(out_path, sep="\t")
def counts(path):
    total_nr = 0
    nr_per_rel = []
    dict_rel = create_dict(path)
    nr_of_rels = len(dict_rel)
    dict_rel_counts = {"Derivation":0, "Inflection":0}
    nr_inf = 0
    nr_der = 0
    for k, v in dict_rel.items():
        nr_per_rel.append((k, len(v)))
        total_nr += len(v)
        if k.startswith("DER"):
            dict_rel_counts["Derivation"] += len(v)
            nr_der +=1
        elif k.startswith("INF"):
            dict_rel_counts["Inflection"] += len(v)
            nr_inf +=1
        else:
            print("wrong relation name?", k)
    dict_rel_counts["Derivation_relations"] = nr_der
    dict_rel_counts["Inflection_relations"] = nr_inf
    return total_nr, nr_of_rels, nr_per_rel, dict_rel_counts


def save_info(dict_info, out_path, keyword):
    with open(out_path, 'w') as wf:
        #wf.write("metadata for: {}\n".format(keyword))
        wf.write("{}\t{}\t{}\n".format("split", "nr of data points", "nr of relations"))
        for k, v in dict_info.items():
            tot_nr = v[0]
            nr_rels = v[1]
            wf.write("{}\t{}\t{}\n".format(k + "_ALL", str(tot_nr), str(nr_rels)))
            dict_rel_counts = v[3]
            wf.write("{}\t{}\t{}\n".format(k +"_Derivation", str(dict_rel_counts["Derivation"]), str(dict_rel_counts["Derivation_relations"])))
            wf.write("{}\t{}\t{}\n".format(k +
                "_Inflection", str(dict_rel_counts["Inflection"]), str(dict_rel_counts["Inflection_relations"])))


def save_relations(dict_info, out_path, keyword):
    with open(out_path, 'w') as wf:
        #wf.write("all relations for: {}\n \n \n".format(keyword))
        wf.write("{}\t{}\t{}\n".format("split", "relation", "nr of data points"))
        for k, v in dict_info.items():
            rels = v[2]
            for r in rels:
                wf.write("{}\t{}\t{}\n".format(k, r[0], r[1]))


def main(path, keyword,type):
    if type == "data_sets":
        out_path = os.path.join(path, "README.txt")
        out_path_rels = os.path.join(path, "relations.txt")
        dict_info = dict()
        for f in os.listdir(path):
            print(f)
            name = f.strip(".csv").strip("combined_")
            print(name)
            path_file = os.path.join(path, f)
            tot_nr, nr_rels, nr_membs, dict_rel_counts = counts(path_file)
            dict_info[name] = (tot_nr, nr_rels, nr_membs, dict_rel_counts)

        save_info(dict_info,  out_path, keyword)
        save_relations(dict_info, out_path_rels, keyword)
    else:
        average_counts(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_directory")
    parser.add_argument("dataset_name")
    parser.add_argument("type")
    args = parser.parse_args()
    main(args.path_directory, args.dataset_name, args.type)
