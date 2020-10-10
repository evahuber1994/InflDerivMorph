from data_reader import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
from utils import create_dict
import numpy as np
import random
def offset_vectors(file_path, embedding_path, emb_dim, out_path):
    dict_rel = create_dict(file_path)
    extractor = FeatureExtractor(embedding_path, emb_dim)
    dict_matrices = dict()

    for k,v in dict_rel.items():
        print(k)
        matr_cur = np.zeros((1000, emb_dim))
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
                        sims.append(cosine_similarity(v[i].reshape(-1,1), v[j].reshape(-1,1)))
            wf.write("{}\t{}\n".format(k, str(np.mean(sims))))


def main():
    path = 'FINAL/DE/normal_o1o/combined_train.csv'
    e_path = 'de_conll_data/model.fifu'
    out_path = 'embedding_measures/de_normalo1o_train.csv'
    avg_sims = offset_vectors(path, e_path, 100, out_path)
    print("avg sims computed")
    with open(out_path, 'w') as wf:
        wf.write("{}\t{}\n".format("relation", "average_offset_similarity"))
        for k,v in avg_sims.items():
            wf.write("{}\t{}\n".format(k, str(v)))

if __name__ == "__main__":
    main()