import pandas as pd
import csv
from data_reader import read_deriv

def map_labels(path_labels, path_in, path_out, separator="\t", type_file='data'):
    old2new = dict()


    with open(path_labels, 'r') as rf:
        next(rf)
        for l in rf:
            l = l.strip()
            line = l.split("\t")
            old2new[line[0]] = line[1]

    if type_file =='data':
        rels, w1, w2 = read_deriv(path_in, splitter="\t")
        with open(path_out, 'w') as wf:
            wf.write("{}\t{}\t{}\n".format("relation", "w1", "w2"))
            for r,wone, wtwo in zip(rels, w1, w2):
                rel = r
                if r in old2new:
                    rel = old2new[r]
                wf.write("{}\t{}\t{}\n".format(rel, wone, wtwo))
    elif type_file == 'embedding':
        with open(path_out, 'w') as wf:
            with open(path_in, 'r') as rf:
                for l in rf:
                    l = l.strip()
                    line = l.split("\t")
                    rel = line[0]
                    if line[0] in old2new:
                        rel = old2new[line[0]]
                    wf.write("{}\t{}\n".format(rel, line[1]))

    else:
        df = pd.read_csv(path_in, delimiter=separator)
        print(df.keys())
        for r in df['relation']:
            if r in old2new:
                df['relation'] = df['relation'].replace([r],old2new[r])
        print(path_out)
        df.to_csv(path_out, sep="\t", index=False)


def read_file(path):
    df = pd.read_csv(path, delimiter="\t")
    indices = df[df['Lemma'].str.contains(" ") == True].index
    df.drop(indices, inplace=True)
    indices = df[df['Suffix_Morph'].isnull()].index
    df.drop(indices, inplace=True)
    df = df.drop_duplicates(subset=['Lemma'])
    df.drop(df[(df['Suffix_Morph'] == "#VALUE!") & (df['Base'] == "#VALUE!")].index, inplace=True)
    df = df.loc[:, df.columns.intersection(['Lemma', 'Base', 'Suffix_Morph', 'Suffix', 'Base_POS', 'Lemma_POS'])]
    return df


def save_df(df, out_path):
    df.to_csv(out_path, sep="\t", index=False)


def make_relations(path, column_nr, threshold =80):
    """
    extract relations from the already preprocessed file, count how many are above a certain threshold
    """
    dict_rel = dict()
    with open(path, 'r') as rf:
        next(rf)
        for l in rf:
            l = l.strip()
            if not l: continue
            line = l.split("\t")
            pos1 = line[4].split("+")[0]
            pos2 = line[5].split("+")[0]
            rel = "{}_{}_{}".format(pos1, pos2, line[column_nr])
            if rel in dict_rel:
                dict_rel[rel].append(line)
            else:
                dict_rel[rel] = [line]
    count = 0
    for k, v in dict_rel.items():
        if len(v) >= threshold:
            count += 1
    return dict_rel, count

def save_correct_format(dict_rel, out_path):
    with open(out_path, 'w') as wf:
        writer = csv.writer(wf, delimiter="\t")
        writer.writerow(("relation_old", "relation", "w1", "w2", "suffix", "allomorph"))
        for k,v in dict_rel.items():
            for w in v:
                spl = k.split("_")
                old_rel = spl[1] + "_" + spl[1] +"_" + spl[-1]
                writer.writerow((old_rel,k, w[1].lower(), w[0].lower(),w[2], w[3]))
def main():

if __name__ == "__main__":
    main()
