import argparse
import pandas as pd

def add_baseline_results(df_baseline, df_results):
    df_baseline.rename(columns={'prec_at_1':'prec_at_1_baseline'}, inplace=True)
    df_baseline.drop(['prec_at_5','prec_at_50','prec_at_80', 'prediction_sim'], axis=1, inplace=True)
    df_results.drop(['prec_at_5','prec_at_50','prec_at_80', 'prediction_sim'], axis=1, inplace=True)
    df_merged = df_results.merge(df_baseline, on='relation')
    return df_merged
def add_information(df_res, df_inf, merge_criteria = "relation", info = "all", info_all=True):
    #df_info_selected = pd.DataFrame()
    #df_inf.drop('relation', axis = 1,inplace=True)
    if info_all:
        df_info_selected = df_inf
        #df_info_selected = df_inf.drop('comments', axis=1)
    else:
        df_info_selected = df_inf[info]

    df_merged = pd.merge(df_res, df_info_selected, on=merge_criteria, how='outer')
    #df_merged.drop_duplicates(inplace=True)
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
    path = '/home/evahu/Documents/Master/Master_Dissertation/results_final/TURKISH/all_rels_normal_means_mapped.csv'
    path_info = "/home/evahu/Documents/Master/Master_Dissertation/results_final/TURKISH/quantitative/Turkish1_resperword_counts_ambiguous_mapped.csv"
    out_path = path.replace(".csv", "_freqs.csv")
    df_res = pd.read_csv(path, delimiter="\t")
    df_info = pd.read_csv(path_info, delimiter="\t")
    print(df_res.keys())
    df_out = add_information(df_res, df_info)
    print(df_out)

    """
    df_baseline = pd.read_csv(path_info, delimiter="\t")
    print(df_baseline.keys())

    df_merged = add_baseline_results(df_baseline, df_res)
    print(df_merged.keys())
    df_merged.to_csv(out_path, sep="\t", index=False)"""
if __name__ == "__main__":
    main()