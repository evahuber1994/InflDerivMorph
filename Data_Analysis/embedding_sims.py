from data_reader import read_deriv, FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np



def get_sim_per_word(path, out_path, emb_path):

    feature_extractor = FeatureExtractor(emb_path, 100)
    rels, w1s, w2s = read_deriv(path)
    with open(out_path, 'w') as wf:
        wf.write("{}\t{}\t{}\t{}\n".format("relation", "w1", "w2", "cos_sim"))
        for w1, w2, r in zip(w1s, w2s, rels):
            w1_emb = feature_extractor.get_embedding(w1)
            w2_emb = feature_extractor.get_embedding(w2)
            sim = cosine_similarity(w1_emb.reshape(1,-1), w2_emb.reshape(1,-1))
            wf.write("{}\t{}\t{}\t{}\n".format(r, w1, w2, str(round(sim[0][0], 2))))

def get_sim_per_relation(in_path):
    out_path = in_path.replace(".csv", "_avg.csv")
    rel_dict = dict()
    with open(in_path, 'r') as rf:
        next(rf)
        for l in rf:
            l = l.strip()
            line = l.split("\t")
            if line[0] not in rel_dict:
                rel_dict[line[0]] = []
            rel_dict[line[0]].append(float(line[3]))
    with open(out_path, 'w') as wf:
        wf.write("{}\t{}\n".format("relation", "average_sim"))
        for k,v in rel_dict.items():
            wf.write("{}\t{}\n".format(k, np.mean(v)))

def main():
    file = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data/FINAL/DE/normal_o1o/combined_train.csv'
    emb_path = '../embeddings/de_45/model.fifu'
    out_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data/FINAL/DE/emb_sims_word.csv'
    #get_sim_per_word(file, out_path, emb_path)
    get_sim_per_relation(out_path)

if __name__ == "__main__":
    main()