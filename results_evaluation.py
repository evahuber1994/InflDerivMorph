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

def plot_results(data_frame, name_column1, name_column2):
    df = data_frame.sort_values(by=[name_column2],ascending=False)
    x = df[name_column1].to_numpy()
    y = df[name_column2].to_numpy()


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

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(out_path)
    plt.show()



def main():
    path1 = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/results_combined/embeddings_new.txt'
    path2 = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/results_combined/results_embs.csv'
    df_embs = embedding_comparison(path1, path2)
    out = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/results_combined/results_top5.csv'
    find_closest(df_embs, out)
    plot_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/results_combined/plot.png'
    tsne_plot(path1, plot_path)
if __name__ == "__main__":
    main()