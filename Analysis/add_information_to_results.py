import argparse
import pandas as pd



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
    path = '/home/evahu/Documents/Master/Master_Dissertation/results_final/GERMAN/category_analysis/results_per_word_normal.csv'
    path_info = "/home/evahu/Documents/Master/Master_Dissertation/results_final/GERMAN/category_analysis/final_analysis_sheet.csv"
    out_path = path.replace(".csv", "_with_counts.csv")
    df_res = pd.read_csv(path, delimiter="\t")
    print(df_res.keys())
    df_info = pd.read_csv(path_info, delimiter="\t")
    print(df_info.keys())

    df_merged = add_information(df_res, df_info, merge_criteria=['relation', 'w2'])
    df_merged.to_csv(out_path, sep="\t", index=False)
if __name__ == "__main__":
    main()