import argparse
import pandas as pd


def add_information(df_res, df_inf, info = "all", info_all=True):
    #df_info_selected = pd.DataFrame()
    if info_all:
        df_info_selected = df_inf
        #df_info_selected = df_inf.drop('comments', axis=1)
    else:
        df_info_selected = df_inf[info]

    df_merged = pd.merge(df_res, df_info_selected, on='relation')
    return df_merged
def main():
    """
    results_path = "/home/evahu/Documents/Master/Master_Dissertation/results_firstround/results/GERMAN_FINAL/normal/german_normal_1/results_per_relation.csv"
    meta_info_path = "/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data/meta_info/GERMAN_DATA_OVERVIEW.csv"
    out_path = "/home/evahu/Documents/Master/Master_Dissertation/results_firstround/results/GERMAN_FINAL/normal/german_normal_1/results_per_relation_with_meta.csv"
    df_results = pd.read_csv(results_path, delimiter="\t", index_col='relation')
    df_info = pd.read_csv(meta_info_path, delimiter=",", index_col='relation')
    print(df_results.keys())
    print(df_info.keys())
    df_merged = add_information(df_results, df_info)
    df_merged.to_csv(out_path, sep="\t")
    """
    results_path = "/home/evahu/Documents/Master/Master_Dissertation/results_firstround/results/GERMAN_FINAL/normal/german_normal_1/results_per_relation.csv"
    meta_path = "/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data/FINAL/DE/emb_sims_word_avg.csv"
    df_res = pd.read_csv(results_path, delimiter="\t")
    df_info =  pd.read_csv(meta_path, delimiter="\t")
    df_m = add_information(df_res, df_info)
    out_path = "/home/evahu/Documents/Master/Master_Dissertation/results_firstround/results/GERMAN_FINAL/normal/german_normal_1/results_per_relation_CS.csv"
    df_m.to_csv(out_path, sep="\t")
if __name__ == "__main__":
    main()