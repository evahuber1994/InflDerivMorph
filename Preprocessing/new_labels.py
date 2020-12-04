"""
creates uniform labels (I1, I2, I3, I..., In)
"""

def create_and_save_labels(in_path, out_path):
    ders = set()
    infs = set()
    with open(in_path, 'r') as rf:
        next(rf)
        for l in rf:
            line = l.split("\t")
            rel = line[1]
            if rel.startswith("DER"):
                ders.add(rel)
            else:
                infs.add(rel)
    infs = sorted(infs)
    ders = sorted(ders)
    print(len(infs), sorted(infs))
    print(len(ders),sorted(ders))
    with open(out_path, 'w') as wf:
        wf.write("{}\t{}\n".format("old", "new"))
        for i, inf in enumerate(infs):
            label = "I" + str(i+1)
            wf.write("{}\t{}\n".format(inf, label))
        for i, der in enumerate(ders):
            label = "D" + str(i+1)
            wf.write("{}\t{}\n".format(der, label))







def main():
    rels_path = "normal_o1o/relations_ru.txt"
    label_path = "LABELS/russian2.csv"
    create_and_save_labels(rels_path, label_path)

if __name__ == "__main__":
    main()
