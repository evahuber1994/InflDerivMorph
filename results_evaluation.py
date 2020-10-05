import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import csv
import seaborn as sns
from sklearn.manifold import TSNE
from data_reader import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity


def read_file(path):
    return pd.read_csv(path, sep="\t")

def offset_vectors(file_path, embedding_path, emb_dim):
    extractor = FeatureExtractor(embedding_path, emb_dim)
    dict_offset = dict()
    with open(file_path, 'r') as f:
        next(f)
        for l in f:
            l = l.strip()
            if not l: continue
            line = l.split("\t")
            relation = line[0]
            w1 = line[1]
            w2 = line[2]
            diff = extractor.get_embedding(w1) - extractor.get_embedding(w2)
            if relation in dict_offset:
                dict_offset[relation].append((w1,w2,diff))
            else:
                dict_offset[relation] = [(w1, w2, diff)]
    #compare mean offset vector euclidean distance
    print(dict_offset.keys())
    """
    dict_sim = dict()
    for k,v in dict_offset.items():
        print(len(v))
       
        sim =0
        for x in v:
            for y in v:
                if x[0] is not y[0] and x[1] is not y[1]:
                    print("hello")
                    sim += cosine_similarity(x[2].reshape(1, -1) , y[2].reshape(1, -1) )
        print(sim/len(v))
        dict_sim[k] = sim/len(v)
    print(dict_sim)
    """



def average_results_derinf(path):
    out_path = path.replace(".csv", "_average.csv")
    df = pd.read_csv(path, delimiter="\t")
    df = df.set_index(['relation'])
    l_relations = df.index.tolist()
    infs = []
    ders = []
    for x in l_relations:
        if ";" in x:
            infs.append(x)
        else:
            ders.append(x)
    df_inf = df.loc[infs, :]
    df_der = df.loc[ders, :]
    mean_acc1_inf = df_inf['acc_at_1'].mean()
    mean_acc5_inf = df_inf['acc_at_5'].mean()
    mean_acc1_der = df_der['acc_at_1'].mean()
    mean_acc5_der = df_der['acc_at_5'].mean()
    with open(out_path, 'w') as wf:
        wf.write("\t r@5\tr@1\n")
        wf.write("inflection\t{}\t{}\n".format(str(round(mean_acc5_inf,2)), str(round(mean_acc1_inf,2))))
        wf.write("derivation\t{}\t{}\n".format(str(round(mean_acc5_der,2)), str(round(mean_acc1_der,2))))


def plot_results(data_frame, name_column1, name_column2, out_path):
    """
    http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
    """
    df = data_frame.sort_values(by=[name_column2],ascending=False)
    x = df[name_column1].to_numpy()
    y = df[name_column2].to_numpy()
    #
    #ax = plt.figure(figsize=(50,10))
    plt.figure(figsize=(20, 13))

    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    #plt.yticks(range(0, 91, 10), [str(x) + "%" for x in range(0, 91, 10)], fontsize=14)
    plt.xticks(fontsize=10)
    plt.xticks(rotation=90)
    plt.plot(x,y)
    plt.text("N;DAT;PL", 1, "Precision at rank 5", fontsize=17, ha="center")
    #plt.text(x, y, s=11)
    #axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    #rect = [0.1, 0.1, 0.8, 0.8]
    #axes = fig.add_axes(rect)
    #axes.yaxis.label.set_size(5)
    #axes.plot(x,y)


    plt.savefig(out_path)
    plt.show()

def embedding_comparison(path_embedding, out_path, write=False):
    extractor =FeatureExtractor(path_embedding, 200)
    voc = extractor.vocab
    relations = [v for v in voc]
    embs = extractor.get_array_embeddings(relations)
    word2word_matrix = cosine_similarity(embs)
    df =  pd.DataFrame(word2word_matrix, index=relations,
                        columns=relations)
    if write:
        df.to_csv(out_path, sep='\t')
    return df
def find_closest(df_embs, out_path):
    #df=df_embs.sort(axis=1, ascending=False)
    dict_closest = dict()
    with open(out_path, 'w') as file:
        writer = csv.writer(file, delimiter = "\t")
        writer.writerow(("relation", "1", "2", "3", "4", "5"))
        for name, row in df_embs.iterrows():
            r = row.sort_values(ascending=False)
            top5 = r[1:6]
            tups = [(x,y) for x,y in zip(top5.values, top5.keys().tolist())]
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
    """    path_embs = "/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/de_results/de_results_conll/embeddings.txt"
    path_out = "/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/french_results/fr_results_conll_first_normal/embedding_NN.csv"
    df = embedding_comparison(path_embs, path_out, write=False)
    find_closest(df, path_out)


    path_embedding = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/french_results/fr_results_conll/embeddings.txt'
    out = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/french_results/fr_results_conll/embeddings_plot.png'
    tsne_plot(path_embedding, out)


    #plot results
    path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/de_results/de_results_conll/results_per_relation.csv'
    out = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/de_results/de_results_conll/plot_results.png'
    df = pd.read_csv(path, sep='\t')
    plot_results(df, 'relation', 'acc_at_5', out)

"""
    path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/all_results/russian_results/ru_results_conll_2nd/results_per_relation.csv'

    average_results_derinf(path)
#path1 = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/results_combined/embeddings_new.txt'
    #path2 = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/results_combined/results_embs.csv'
    #df_embs = embedding_comparison(path1, path2)


    #out = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/results_combined/results_top5.csv'
    #find_closest(df_embs, out)
    #plot_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/results_combined/plot.png'
    #tsne_plot(path1, plot_path)

    #emb_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/embeddings/word2vec-mincount-30-dims-100-ctx-10-ns-5.w2v'
    #file_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data2/combined_test.csv'
    #offset_vectors(file_path, emb_path, 200)

if __name__ == "__main__":
    main()