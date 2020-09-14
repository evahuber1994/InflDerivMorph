import pandas as pd
import os

def nr_instances(file_path):
    df = pd.read_csv(file_path, delimiter='\t')
    return df.shape[0]


def main():
    """
    dir = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data2/DERIVATION/relation_files'
    out = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data2/DERIVATION/nr_relations.csv'
    dict_rels = dict()
    for d in os.listdir(dir):
        path = os.path.join(dir, d)
        rel = d.strip(".csv")
        nr = nr_instances(path)
        dict_rels[rel] = nr
    with open(out, 'w') as file:
        file.write("{}\t{}\n".format("relation", "nr"))
        for k,v in dict_rels.items():
            file.write("{}\t{}\n".format(k,v))

    out = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data2/DERIVATION/nr_relations.csv'
    results = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/results/all_summaries'
    new_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/results/comb.csv'
    df = pd.read_csv(out, delimiter='\t')
    for file in os.listdir(results):
        path = os.path.join(results, file)
        name = file.strip('summary_')
        name = name.strip('_deriv.csv')
        new_rank_name = "avg_rank_" + name
        df2 = pd.read_csv(path, delimiter= '\t')
        df2 = df2.rename(columns = {"avg_rank": new_rank_name})
        df2 = df2.drop("avg_reciprank", axis=1)
        df2 = df2.drop("avg_sim", axis=1)
        df = pd.merge(df2, df)
    df.to_csv(new_path, sep='\t')
    #df_comb = pd.merge(df1, df2)
      """

if __name__ == "__main__":
    main()