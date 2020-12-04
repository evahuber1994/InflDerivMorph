from data_reader import read_deriv
from utils import create_dict
import csv
"""
removes one-to-one transformations
"""



def check_1t1(dict_rel):
    dict_one2ones = {k:[] for k in dict_rel.keys()}
    one2ones = []
    dict_lengths = dict()
    for k,v in dict_rel.items():
        the_same = True
        dict_lengths[k] = len(v)
        for w in v:
            if w[0] == w[1]:
                dict_one2ones[k].append((w[0], w[1]))
            else:
                the_same = False
        if the_same:
            one2ones.append(k)
    return dict_one2ones, one2ones, dict_lengths

def write_smaller_file(dict_rel, o2o, out_path):
    with open(out_path, 'w') as wf:
        writer = csv.writer(wf, delimiter="\t")
        writer.writerow(("relation", "w1", "w2"))
        for k,v in dict_rel.items():
            if k not in o2o:
                for w in v:
                    writer.writerow((k, w[0], w[1]))

def main():

    path = "ALL_WITH_NOUNSthresh80.csv"
    out_path = path.replace(".csv", "_no1t1.csv")
    dict_rel = create_dict(path)

    dict_1t1, o2o, d_lengths = check_1t1(dict_rel)
    for k,v in dict_1t1.items():
        if (d_lengths[k] - len(v)) < 10:
            if k not in o2o:
                o2o.append(k)
    write_smaller_file(dict_rel, o2o, out_path)



if __name__ == "__main__":
    main()

