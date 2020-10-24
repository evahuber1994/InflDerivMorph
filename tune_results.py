import os
import pandas as pd
dir_path = '/home/eva/master_diss/results/tune/french'
out_path = '/home/eva/master_diss/results/tune/french/results_tune_french.csv'
out_path2 = out_path.replace(".csv", "_summary.csv")
all = pd.DataFrame()
all2 = pd.DataFrame()
for d in os.listdir(dir_path):
    tune_settings = d
    path_subd = os.path.join(dir_path,d)
    df_1 = pd.DataFrame()
    df_2 = pd.DataFrame()
    for f in os.listdir(path_subd):
        path = os.path.join(path_subd,f)
        if f == 'results_per_relation_average.csv':
            df_sub = pd.read_csv(path, delimiter="\t")
            df_1 = df_sub.iloc[:,0:5]
            df_1.insert(0, 'tune_settings', [tune_settings, tune_settings])
        elif f == 'summary.csv':
            df_sub = pd.read_csv(path, delimiter="\t")
            df_2 = df_sub.iloc[:,0:4]
            df_2.insert(0, 'tune_settings', [tune_settings])

        else:
            continue

    prev = pd.concat([all, df_1])
    all = prev
    prev2 = pd.concat([all2, df_2])
    all2 = prev2
all.to_csv(out_path, sep="\t", index=False)
all2.to_csv(out_path2, sep="\t", index=False)
