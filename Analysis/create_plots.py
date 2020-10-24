import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.lines as mlines

def rank_distributions(df, out_path): #relation targetword rank
    relations = list(df['relation'])
    numb_bins = 100
    nr_relations = len(relations)
    nrows = 18
    ncols = 7
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,figsize=(30,30))

    count = 0
    for i in range(0,nrows):
        for j in range(0, ncols):
            if count > 121:
                break
            sub_df = df[df['relation'] == relations[count]]
            upper_value = sub_df.shape[0]
            ax[i, j].hist(sub_df['rank'], numb_bins, range=(0, 100),facecolor='blue')
            ax[i, j].set_title(relations[count])
            count += 1
    fig.tight_layout()
    plt.savefig(out_path + "all_in_one.pdf")
    plt.show()

    """
    for r in relations:
        
        path_figure = out_path + r + ".pdf"
        sub_df = df[df['relation'] == r]
        print(sub_df['rank'].shape)
        upper_value = sub_df.shape[0]
        plt.hist(sub_df['rank'], numb_bins, range=(0, upper_value),facecolor='blue')
        plt.savefig(path_figure)
    """

def scatter_rank_against_frequency(df, path_out):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1, )
    #plt.gca().invert_yaxis()
    #plt.gca().invert_xaxis()
    counts_log = np.log2(df["count"])
    ax.scatter(df["rank"], counts_log , s=4)
    plt.show()


def scatter_value_against_performance(df,  x_value, y_value, x_label, y_label, path_out, regression_line=False, log2=False):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, )

    df_inf = df[df["relation"].str.startswith("INF")]
    df_der = df[df["relation"].str.startswith("DER")]
    print(df_inf.shape)
    print(df_der.shape)
    rel_label_inf = "Inflection"
    rel_label_der = "Derivation"
    x_inf = df_inf[x_value]
    y_inf = df_inf[y_value]

    x_der = df_der[x_value]
    y_der = df_der[y_value]

    if log2:
        y_inf = np.log2(y_inf)
        y_der = np.log2(y_der)
    ax.scatter(x_inf, y_inf, label=rel_label_inf, s=10, c="r")
    ax.scatter(x_der, y_der, label=rel_label_der, s=10, c="g")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


    ax.legend()
    ax.spines["top"].set_color("None")
    ax.spines["right"].set_color("None")
    if regression_line:
        m, b = np.polyfit(df[x_value], df[y_value], 1)
        plt.plot(df[x_value], m * df[x_value] + b)
    fig.tight_layout()
    plt.savefig(path_out)
    plt.show()




def change_in_rank(df_in, out_path, nr_rows=30):
    """
    df["color"] = df.apply(lambda row: "green" if row["1957"] >= row["1952"] else "red", axis = 1)
    https://www.kaggle.com/python10pm/plotting-with-python-learn-80-plots-step-by-step
    :param data_frame:
    :param out_path:
    :return:
    """
    rows = nr_rows

    fig, ax = plt.subplots(1, 1, figsize=(10, 20))
    df_in.sort_values('Rank_x', axis=0, ascending=True, inplace=True)
    print(df_in.shape)
    df = df_in
    if nr_rows != 0:
        df = df_in[:nr_rows]  # .head(nr_rows)


    print(df.head())
    rels = list(df['relation'])
    print(len(rels), rels)
    ranks_X = df["Rank_x"].tolist()
    ranks_Y = df["Rank_y"].tolist()
    scale = len(rels)
    print(rels)
    # write the vertical lines
    ax.vlines(x=0, ymin=1, ymax=scale, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
    ax.vlines(x=3, ymin=1, ymax=scale, color='black', alpha=0.7, linewidth=1, linestyles='dotted')

    #school_y_vals = np.linspace(1, scale - 1, num=len(school_list))
    #major_y_vals = np.linspace(1, scale - 1, num=len(major_list))
    #ax.scatter(y=school_y_vals, x=np.repeat(1, len(school_list)), s=10, color='black', alpha=0.7)
    #ax.scatter(y=major_y_vals, x=np.repeat(3, len(major_list)), s=10, color='black', alpha=0.7)

    # write the lines and annotation
    rank_vals_x = sorted(ranks_X)
    rank_vals_y = sorted(ranks_Y)


    ax.scatter(y=rank_vals_x, x=np.repeat(0, len(set(rels))), s=10, color='black', alpha=0.7)
    ax.scatter(y=rank_vals_y, x=np.repeat(3, len(set(rels))), s=10, color='black', alpha=0.7)

    # write the lines and annotation
    plt.gca().invert_yaxis()
    for r, x, y in zip(rels, ranks_X, ranks_Y):
        print(x,y, r)
        # ax.text(1 - 0.05, y_val. r, horizontalalignment='right', verticalalignment='center',
        #        fontdict={'size': 10})
        colour = ""
        ax.text(0 - 0.05, x, r, horizontalalignment='right', verticalalignment='center', fontdict={'size': 10})
        if x > y:
            colour = "r"
        elif y > x:
            colour = "g"
        else:
            colour = "b"

        newline([0, x], [3, y], color=colour)


        #ax.set(xlim=(0, 4), ylim=(0, scale))
    ax.axis('off')

    plt.title("rank changes")
    plt.savefig(out_path)
    plt.show()

        # df.sort_values('Rank_x', axis=0, ascending=False, inplace=True)

        # df["color"] = df.apply(lambda row: "green" if row["Rank_x"] >= row["Rank_y"] else "red", axis=1)
        # fig = plt.figure(figsize=(8, 12))
        # ax = fig.add_subplot()
        # for r in df['relation']:
        #     start_x = df[df['relation'] == r]['Rank_x']
        #     finish_x = df[df['relation'] == r]['Rank_y']
        #     colour = df[df['relation'] == r]['color']


def newline(p1, p2, color='black'):
    ax = plt.gca()
    l = mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]], color=color, marker='o', markersize=6)
    ax.add_line(l)
    return l


def plot_results(data_frame, name_column1, name_column2, out_path):
    """
    http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
    """
    df = data_frame.sort_values(by=[name_column2], ascending=True)

    fig, ax = plt.subplots(figsize=(20, 20))
    rels = list(df['relation'])
    colours = []
    cmap = plt.cm.coolwarm
    for r in rels:
        if r.startswith("D"):
            colours.append(cmap(.9))
        else:
            colours.append(cmap(0.3))
    ax.barh(df[name_column1], df[name_column2], color=colours)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # ax.legend()
    ax.set_title("Precision at rank 1")

    plt.xlim(0, 1.0)
    plt.ylim(rels[0], rels[-1])
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()



def create_scatter_plot(x_axis, y_axis, df):
    # df.plot.scatter(x=x_axis, y=y_axis)
    # plt.show()
    df_sub = df[df['relation'].str.startswith("I")]
    df_sub2 = df[df['relation'].str.startswith("D")]
    plt.scatter(df_sub['prec_at_1'], df_sub[x_axis], marker='^')
    plt.scatter(df_sub2['prec_at_1'], df_sub2[x_axis], marker='o')

    plt.show()


def main():
    """
    path = '/home/evahu/Documents/Master/Master_Dissertation/results_final/GERMAN/mean_relations_small_means.csv'
    out = '/home/evahu/Documents/Master/Master_Dissertation/results_final/GERMAN/mean_relations_small_means_plot1.pdf'
    #df = pd.read_csv(path, sep='\t', index_col='relation')
    df = pd.read_csv(path, sep='\t')
    print(df.head())
    plot_results(df, 'relation', 'prec_at_1', out)
    """
    path = '/home/evahu/Documents/Master/Master_Dissertation/results_final/GERMAN/mean_relations_rank_plustraining.csv'
    out_path = '/home/evahu/Documents/Master/Master_Dissertation/results_final/GERMAN/mean_relations_rank_plustraining_plot_whole.pdf'
    df = pd.read_csv(path, delimiter="\t")
    change_in_rank(df, out_path, nr_rows=0)
    """
    path_in = "/home/evahu/Documents/Master/Master_Dissertation/results_final/GERMAN/mean_relations_normal_means_with_offsets.csv"
    path_out = path_in.replace(".csv", "_plot.pdf")
    df = pd.read_csv(path_in, delimiter="\t")
    print(df.keys())

    y_axis = "average_offset_similarity"
    scatter_value_against_performance(df, "prec_at_1", y_axis, "performance", "average_offset_similarity", path_out, log2=False)
  
    path = '/home/evahu/Documents/Master/Master_Dissertation/results_final/GERMAN/results_per_word_normal.csv'
    df = pd.read_csv(path, delimiter="\t")
    out_path = "/home/evahu/Documents/Master/Master_Dissertation/results_final/GERMAN/rank_distributions/"
    rank_distributions(df, out_path)
    """
if __name__ == "__main__":
    main()
