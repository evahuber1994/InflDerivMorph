import pandas as pd
import csv

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
            pos1 = line[5].split("+")[0]
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
        writer.writerow(("relation", "w1", "w2", "suffix", "allomorph"))
        for k,v in dict_rel.items():
            for w in v:
                writer.writerow((k, w[1].lower(), w[0].lower(),w[2], w[3]))
def main():
    # path = '/home/evahu/Documents/Master/Master_Dissertation/data/Turkish/TrLex.csv'
    # df = read_file(path)
    # save_correct_format(df, out_path)
    path = '/home/evahu/Documents/Master/Master_Dissertation/data/Turkish/TrLex_small.csv'
    dic, co = make_relations(path, column_nr=3)
    print(co)
    out_path = '/home/evahu/Documents/Master/Master_Dissertation/data/Turkish/Turkish_Derivation_prepared_allomorphs.csv'
    save_correct_format(dic, out_path)
if __name__ == "__main__":
    main()
