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

def add_ranks_small_big(path_big, path_small, out_path):
    df_big = pd.read_csv(path_big, delimiter=",")
    df_small = pd.read_csv(path_small,  delimiter="\t")

    #df_big.sort_values('prec_at_1', axis=0, ascending=False, inplace=True)
    #df_small.sort_values('prec_at_1', axis=0,ascending=False, inplace=True)
    df_big["Rank"] = df_big['prec_at_1'].rank(ascending=False)
    df_small["Rank"] = df_small['prec_at_1'].rank(ascending=False)
    df_merge = pd.merge(df_big, df_small, on='relation')
    df_merge.sort_values('Rank_x', axis=0, ascending=True, inplace=True)
    df_merge.to_csv(out_path, sep="\t")

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


def get_mean_relations(dir_path, out_path):
    dfs = []
    for subd in os.listdir(dir_path):
        path_subd = os.path.join(dir_path, subd)
        for f in os.listdir(path_subd):
            if f == "results_per_relation.csv":
                path = os.path.join(path_subd, f)
                new_df = pd.read_csv(path, delimiter="\t", index_col='relation')
                print(new_df.shape)
                dfs.append(new_df)

    df_concat = pd.concat(dfs)
    all_relations = set(df_concat.index.values)
    column_names = []

    df_std = pd.DataFrame( columns=df_concat.keys(), index=all_relations)
    df_mean = pd.DataFrame(columns = df_concat.keys(), index=all_relations)

    for r in all_relations:
        results = df_concat.loc[r]
        mean = results.mean()
        std = results.std()
        df_std.loc[r] = std
        df_mean.loc[r] = mean
        print(df_std)
    out_path_mean = out_path.replace(".csv", "_means.csv")
    out_path_std = out_path.replace(".csv", "_std.csv")
    df_std.to_csv(out_path_std, sep="\t")
    df_mean.to_csv(out_path_mean, sep="\t")




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


def tune_results(dir_path): #tune parameter, measure
    """
    compares differences in results across all tuning settings, takes parameter of what to compare
    """
    paths = ['german_tune_1+True+cosine_distance+sigmoid', 'german_tune_2+True+cosine_distance+sigmoid',
             'german_tune_3+True+cosine_distance+sigmoid']
    tune_parameter = "prec_at_1"
    column_names = paths.insert(0, 'relation')
    final_df = pd.DataFrame(columns=column_names)
    for subd in os.listdir(dir_path):
        #hyperparams = subd.split("+")
        """
        layers = hyperparams[0]
        non_l = hyperparams[1]
        type_non_l = hyperparams[2]
        loss = hyperparams[3]
        """
        if subd in paths:
            subd_path = os.path.join(dir_path, subd)

            for f in os.listdir(subd_path):
                if f == "results_per_relation.csv":
                    f_path = os.path.join(subd_path, f)
                    df = pd.read_csv(f_path, delimiter="\t", index_col='relation')
                    df_partial = df[tune_parameter]
                    final_df[subd] = df_partial
    return final_df



def plot_tune_results(df):
    colors = ['#2300A8', '#00A658']

    plt.figure()


    #df.T.plot.line(legend=False,x_compat=True)
    for i, row in df.iterrows():
        if row.name == "I":
            cr = "r"
            plt.plot(row, color=cr, linewidth=0.5)
        else:
            cr = "g"
        #plt.plot(row, color=cr,linewidth=0.5)
    plt.show()
    #df = df.T.plot()
    #plt.show()
    # with pd.plotting.plot_params.use('x_compat', True):
    #     df['A'].plot(color='r')
    #     df['B'].plot(color='g')
    #     df['C'].plot(color='b')

def main():
    """
    dir_path = "/home/evahu/Documents/Master/Master_Dissertation/results_final/FRENCH2/french_multiruns_small"
    out_path = "/home/evahu/Documents/Master/Master_Dissertation/results_final/FRENCH2/relations_small.csv"
    get_mean_relations(dir_path, out_path)

    path_embs = "/home/evahu/Documents/Master/Master_Dissertation/results_final/Russian/normal/russian_normal_1/embeddings.txt"
    path_out = path_embs.replace(".txt", "_plot.pdf")
    #df = embedding_comparison(path_embs, path_out, write=False)
    #find_closest(df, path_out)

    #out = '/home/evahu/Documents/Master/Master_Dissertation/results_final/TURKISH/turkish_multiruns/turkish_1/embedding_plot.pdf'
    tsne_plot(path_embs, path_out)


    #(path, emb_path, out_path, average=True

    dirrec = '/home/evahu/Documents/Master/Master_Dissertation/results_firstround/results/tune'
    dir_out = '/home/evahu/Documents/Master/Master_Dissertation/results_firstround/results/tune_out_try.csv'
    df = tune_results(dirrec)
    print(df.keys())
    names = df.index.values
    new_names = {old_n: old_n[0] for old_n in names}
    df.rename(new_names, inplace=True)

    #df.to_csv(dir_out, sep="\t")
    print(df.head)
    plot_tune_results(df)

    path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data/FINAL/DE/normal_o1o/combined_test.csv'
    emb_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/embeddings/de_45/model.fifu'
    out_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data/FINAL/DE/normal_o1o/combined_test_embedding.csv'
    cosine_similarity_base_target(path, emb_path, out_path)
    """
    dir_path = '/home/evahu/Documents/Master/Master_Dissertation/results_final/Russian/small'
    out_path = '/home/evahu/Documents/Master/Master_Dissertation/results_final/Russian/all_relations_small.csv'
    get_mean_relations(dir_path, out_path)
    #get_mean_relations(dir_path, out_path)
    """
    #path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data/FINAL/DE/normal/combined_test.csv'
    #emb_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/embeddings/de_45/model.fifu'
    #out_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/analysis/try.csv'
    #cosine_similarity_base_target(path, emb_path, out_path)
    """

  
    #plot results

   

    #path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/all_results/russian_results/ru_results_conll_2nd/results_per_relation.csv'

    #average_results_derinf(path)

    """
    #path1 = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/results_combined/embeddings_new.txt'
    #path2 = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/results_combined/results_embs.csv'
    # df_embs = embedding_comparison(path1, path2)
    
    
    # out = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/results_combined/results_top5.csv'
    # find_closest(df_embs, out)
    # plot_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/results_combined/plot.png'
    # tsne_plot(path1, plot_path)

    # emb_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/embeddings/word2vec-mincount-30-dims-100-ctx-10-ns-5.w2v'
    # file_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data2/combined_test.csv'
    # offset_vectors(file_path, emb_path, 200)


    path = '/home/evahu/Documents/Master/Master_Dissertation/results_final/GERMAN/mean_relations_normal_means.csv'
    path_small ='/home/evahu/Documents/Master/Master_Dissertation/results_final/GERMAN/mean_relations_small_means.csv'
    path_out = '/home/evahu/Documents/Master/Master_Dissertation/results_final/GERMAN/mean_relations_ranks.csv'
    add_ranks_small_big(path, path_small, path_out)

    dir_path = "/home/evahu/Documents/Master/Master_Dissertation/results_final/FRENCH2/french_multiruns_small"
    out_path = "/home/evahu/Documents/Master/Master_Dissertation/results_final/FRENCH2/all_small.csv"
    get_SD(dir_path, out_path)
    """

if __name__ == "__main__":
    main()
