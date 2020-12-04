from data_reader import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
from utils import create_dict
import numpy as np
import random
def offset_vectors(file_path, embedding_path, emb_dim, out_path, threshold):
    dict_rel = create_dict(file_path)
    extractor = FeatureExtractor(embedding_path, emb_dim)
    dict_matrices = dict()

    for k,v in dict_rel.items():
        length = len(v)
        if threshold > 0:
            if len(v) > threshold:
                length = threshold
        print(k)
        matr_cur = np.zeros((length, emb_dim))
        random.shuffle(v)
        for i, w in enumerate(v[:1000]):
            matr_cur[i] =np.subtract(extractor.get_embedding(w[0]), extractor.get_embedding(w[1]))
        dict_matrices[k] = matr_cur
        print(len(dict_matrices))
    print("finished creating offset vector dictionary")
    with open(out_path, 'w') as wf:
        wf.write("{}\t{}\n".format("relation", "average_offset_similarity"))
        for k,v in dict_matrices.items():

            print(k)
            print(v.shape)
            sims = []
            for i in range(v.shape[0]):
                for j in range(v.shape[0]):
                    if i != j:
                        cs = cosine_similarity(v[i].reshape(1,-1), v[j].reshape(1,-1))
                        sims.append(cs[0][0])

            wf.write("{}\t{}\n".format(k, str(np.mean(sims))))
def embedding_sims(path, e_path, e_dim, out_path, average=True):
    dict_rel = create_dict(path)
    avg_sims = {k: [] for k in dict_rel.keys()}
    extractor = FeatureExtractor(e_path, e_dim)
    with open(out_path, 'w') as wf:
        wf.write("{}\t{}\t{}\t{}\n".format("relation", "w1", "w2", "sim"))
        for k,v in dict_rel.items():
            for w in v:
                sim = cosine_similarity(extractor.get_embedding(w[0]).reshape(1,-1), extractor.get_embedding(w[1]).reshape(1,-1))
                sim = round(sim[0][0],3)
                avg_sims[k].append(sim)
                wf.write("{}\t{}\t{}\t{}\n".format(k, w[0], w[1], str(sim)))
    if average:
        out_path2 = out_path.replace(".csv", "_average.csv")
        with open(out_path2, 'w') as wf:
            wf.write("{}\t{}\n".format("relation", "average_sim"))
            for k,v in avg_sims.items():
                avg = np.mean(v)
                wf.write("{}\t{}\n".format(k, str(round(avg,3))))




def main():
    path = 'quantitative_russian/ALL_IN_ONE.csv'
    e_path = 'ru_conll_data/model.fifu'
    out_path = 'quantitative_russian/russian_emb_sims.csv'
    threshold = 1000
    #(file_path, embedding_path, emb_dim, out_path, threshold):
    embedding_sims(path, e_path, 100, out_path)

if __name__ == "__main__":
    main()