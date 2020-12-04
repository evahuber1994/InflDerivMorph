import argparse
import pandas as pd

def add_baseline_results(df_baseline, df_results):
    df_baseline.rename(columns={'prec_at_1':'prec_at_1_baseline'}, inplace=True)
    df_baseline.drop(['prec_at_5','prec_at_50','prec_at_80', 'prediction_sim'], axis=1, inplace=True)
    df_results.drop(['prec_at_5','prec_at_50','prec_at_80', 'prediction_sim'], axis=1, inplace=True)
    df_merged = df_results.merge(df_baseline, on='relation')
    return df_merged

def add_information(df_res, df_inf, merge_criteria = "relation", info = "all", info_all=True):

    if info_all:
        df_info_selected = df_inf
    else:
        df_info_selected = df_inf[info]

    df_merged = pd.merge(df_res, df_info_selected, on=merge_criteria, how='outer')

    return df_merged
def main():

    path = ''
    path_info = ''
    out_path = path.replace(".csv", "_freqs.csv")
    df_res = pd.read_csv(path, delimiter="\t")
    df_info = pd.read_csv(path_info, delimiter="\t")
    df_out = add_information(df_res, df_info)
    print(df_out.keys())

if __name__ == "__main__":
    main()