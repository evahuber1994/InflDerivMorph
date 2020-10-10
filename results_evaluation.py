import pandas as pd
import os
from matplotlib import pyplot as plt
import torch
import csv
import seaborn as sns
from sklearn.manifold import TSNE
from data_reader import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
from utils import create_dict
import numpy as np
import os

def get_SD(dir_path, out_path):
    dict_results_inf = dict()
    dict_results_der = dict()
    for d in os.listdir(dir_path):
        d_path = os.path.join(dir_path, d)
        for f in os.listdir(d_path):
            if f == 'results_per_relation_average.csv':
                f_path = os.path.join(d_path, f)
                df = pd.read_csv(f_path, delimiter="\t")
                df = df.set_index('Unnamed: 0')
                inflection = df.iloc[0]
                derivation = df.iloc[1]
                for i, val in inflection.items():
                    if i not in dict_results_inf:
                        dict_results_inf[i] = [val]
                    else:
                        dict_results_inf[i].append(val)
                for i, val in derivation.items():
                    if i not in dict_results_der:
                        dict_results_der[i] = [val]
                    else:
                        dict_results_der[i].append(val)
    d_inf_final= dict()
    d_der_final= dict()
    print(dict_results_der)
    print(dict_results_inf)
    for k, v in dict_results_inf.items():
        mean_v = np.mean(v)
        sd_v = np.std(v)
        d_inf_final[k] = (mean_v, sd_v)
    for k,v in dict_results_der.items():
        mean_v = np.mean(v)
        sd_v = np.std(v)
        d_der_final[k] = (mean_v, sd_v)

    with open(out_path, 'w') as wf:
        #wf.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format("relation","r1", "sd_r1", "r5", "sd_r5", "r50", "sd_r50", "r80", "sd_r80", "cs", "sd_cs"))
        wf.write("{}\t{}\t{}\t{}\n".format("type", "measure", "mean", "std"))
        for k,v in d_inf_final.items():
            wf.write("{}\t{}\t{}\t{}\n".format("inflection", k, str(v[0]), str(v[1])))
        for k,v in d_der_final.items():
            wf.write("{}\t{}\t{}\t{}\n".format("derivation", k, str(v[0]), str(v[1])))





def read_file(path):
    return pd.read_csv(path, sep="\t")



def cosine_similarity_base_target(path, emb_path, out_path, average=True):
    dict_rel = create_dict(path)
    extractor = FeatureExtractor(emb_path, 100)
    dict_sim = dict()
    with open(out_path, 'w') as wf:
        wf.write("{}\t{}\t{}\t{}\n".format("relation", "word1", "word2", "similarity"))
        for k,v in dict_rel.items():
            dict_sim[k] = []
            for w in v:
                w1 = extractor.get_embedding(w[0])
                w2 = extractor.get_embedding(w[1])
                sim = cosine_similarity(w1.reshape(1,-1), w2.reshape(1,-1))
                wf.write("{}\t{}\t{}\t{}\n".format(k, w[0], w[1], str(round(sim[0][0],2))))
                dict_sim[k].append(sim)
    if average:
        out_path2 = out_path.replace(".csv", "_average.csv")
        with open(out_path2, 'w') as wf:
            wf.write("{}\t{}\n".format("relation", "average_cs"))
            for k,v in dict_sim.items():
                avg = np.mean(v)
                wf.write("{}\t{}\n".format(k, str(round(avg, 2))))



def average_results_derinf(path):
    out_path = path.replace(".csv", "_average.csv")
    df = pd.read_csv(path, delimiter="\t")
    df = df.set_index(['relation'])
    l_relations = df.index.tolist()
    infs = []
    ders = []
    for x in l_relations:
        if x.startswith("INF"):
            infs.append(x)
        else:
            ders.append(x)
    df_inf = df.loc[infs, :]
    df_der = df.loc[ders, :]
    mean_acc1_inf = df_inf['prec_at_1'].mean()
    mean_acc5_inf = df_inf['prec_at_5'].mean()
    mean_acc50_inf = df_inf['prec_at_50'].mean()
    mean_acc80_inf = df_inf['prec_at_80'].mean()
    mean_cos_inf = df_inf['prediction_sim'].mean()
    mean_acc1_der = df_der['prec_at_1'].mean()
    mean_acc5_der = df_der['prec_at_5'].mean()
    mean_acc50_der = df_der['prec_at_50'].mean()
    mean_acc80_der = df_der['prec_at_80'].mean()
    mean_cos_der = df_der['prediction_sim'].mean()
    print(mean_acc1_der, mean_acc1_inf, mean_acc5_der, mean_acc5_inf, mean_acc50_der, mean_acc50_inf)
    print(mean_acc80_der, mean_acc80_inf, mean_cos_der, mean_cos_inf)
    with open(out_path, 'w') as wf:
        wf.write("\tr@1\tr@5\tr@50\tr@80\tcosine_sim\n")
        wf.write("inflection\t{}\t{}\t{}\t{}\t{}\n".format(str(round(mean_acc1_inf, 2)), str(round(mean_acc5_inf, 2)),
                                               str(round(mean_acc50_inf, 2)),
                                               str(round(mean_acc80_inf, 2)),
                                               str(round(mean_cos_inf, 2))))
        wf.write("derivation\t{}\t{}\t{}\t{}\t{}\n".format(str(round(mean_acc1_der, 2)), str(round(mean_acc5_der, 2)),
                                               str(round(mean_acc50_der, 2)),
                                               str(round(mean_acc80_der, 2)),
                                               str(round(mean_cos_der, 2))))


def plot_results(data_frame, name_column1, name_column2, out_path):
    """
    http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
    """
    df = data_frame.sort_values(by=[name_column2], ascending=False)
    x = df[name_column1].to_numpy()
    y = df[name_column2].to_numpy()
    #
    # ax = plt.figure(figsize=(50,10))
    plt.figure(figsize=(20, 13))

    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # plt.yticks(range(0, 91, 10), [str(x) + "%" for x in range(0, 91, 10)], fontsize=14)
    plt.xticks(fontsize=10)
    plt.xticks(rotation=90)
    plt.plot(x, y)
    plt.text("N;DAT;PL", 1, "Precision at rank 5", fontsize=17, ha="center")
    # plt.text(x, y, s=11)
    # axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # rect = [0.1, 0.1, 0.8, 0.8]
    # axes = fig.add_axes(rect)
    # axes.yaxis.label.set_size(5)
    # axes.plot(x,y)

    plt.savefig(out_path)
    plt.show()


def embedding_comparison(path_embedding, out_path, write=False):
    extractor = FeatureExtractor(path_embedding, 200)
    voc = extractor.vocab
    relations = [v for v in voc]
    embs = extractor.get_array_embeddings(relations)
    word2word_matrix = cosine_similarity(embs)
    df = pd.DataFrame(word2word_matrix, index=relations,
                      columns=relations)
    if write:
        df.to_csv(out_path, sep='\t')
    return df


def find_closest(df_embs, out_path):
    # df=df_embs.sort(axis=1, ascending=False)
    dict_closest = dict()
    with open(out_path, 'w') as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(("relation", "1", "2", "3", "4", "5"))
        for name, row in df_embs.iterrows():
            r = row.sort_values(ascending=False)
            top5 = r[1:6]
            tups = [(x, y) for x, y in zip(top5.values, top5.keys().tolist())]
            tups.insert(0, name)
            writer.writerow(tups)
            dict_closest[name] = (top5.values, top5.keys().tolist())


def tsne_plot(path_embedding, out_path):
    "Creates and TSNE model and plots it"
    # taken from https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne
    labels = []
    tokens = []
    extractor = FeatureExtractor(path_embedding, 200)
    for word in extractor.vocab:
        tokens.append(extractor.get_embedding(word))
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(22, 22))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(4, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(out_path)
    plt.show()


def main():
    #(path, emb_path, out_path, average=True
    path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data/FINAL/DE/normal_o1o/combined_test.csv'
    emb_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/embeddings/de_45/model.fifu'
    out_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data/FINAL/DE/normal_o1o/combined_test_embedding.csv'
    cosine_similarity_base_target(path, emb_path, out_path)
    """
    dir_path = '/home/evahu/Documents/Master/Master_Dissertation/results_firstround/results/normal_10'
    out_path = '/home/evahu/Documents/Master/Master_Dissertation/results_firstround/results/normal_10/mean_std.csv'
    get_SD(dir_path, out_path)
  
    #path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data/FINAL/DE/normal/combined_test.csv'
    #emb_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/embeddings/de_45/model.fifu'
    #out_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/analysis/try.csv'
    #cosine_similarity_base_target(path, emb_path, out_path)
    path_embs = "/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/de_results/de_results_conll/embeddings.txt"
    path_out = "/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/french_results/fr_results_conll_first_normal/embedding_NN.csv"
    df = embedding_comparison(path_embs, path_out, write=False)
    find_closest(df, path_out)


    path_embedding = '/home/evahu/Documents/Master/Master_Dissertation/results_firstround/results/FR1/embeddings.txt'
    out = '/home/evahu/Documents/Master/Master_Dissertation/results_firstround/results/FR1/embeddings_plot.png'
    tsne_plot(path_embedding, out)

    
    #plot results
    path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/de_results/de_results_conll/results_per_relation.csv'
    out = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/de_results/de_results_conll/plot_results.png'
    df = pd.read_csv(path, sep='\t')
    plot_results(df, 'relation', 'acc_at_5', out)

    """
    #path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/all_results/russian_results/ru_results_conll_2nd/results_per_relation.csv'

    #average_results_derinf(path)


# path1 = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/results_combined/embeddings_new.txt'
# path2 = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/results_combined/results_embs.csv'
# df_embs = embedding_comparison(path1, path2)


# out = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/results_combined/results_top5.csv'
# find_closest(df_embs, out)
# plot_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/results_combined/plot.png'
# tsne_plot(path1, plot_path)

# emb_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/embeddings/word2vec-mincount-30-dims-100-ctx-10-ns-5.w2v'
# file_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data2/combined_test.csv'
# offset_vectors(file_path, emb_path, 200)

if __name__ == "__main__":
    main()
