import pandas as pd
from matplotlib import pyplot as plt

def create_plot(x_axis, y_axis, df):
    #df.plot.scatter(x=x_axis, y=y_axis)
    #plt.show()
    df_sub = df[df['relation'].str.startswith("I")]
    df_sub2 = df[df['relation'].str.startswith("D")]
    plt.scatter(df_sub['prec_at_1'], df_sub[x_axis], marker='^')
    plt.scatter(df_sub2['prec_at_1'], df_sub2[x_axis], marker='o')

    plt.show()
def main():
    path = "/home/evahu/Documents/Master/Master_Dissertation/results_firstround/results/GERMAN_FINAL/normal/german_normal_1/results_per_relation_CS.csv"
    df = pd.read_csv(path, delimiter="\t")
    print(df.keys())
    create_plot('average_sim', 'prec_at_1', df)
if __name__ == "__main__":
    main()