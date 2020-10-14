from rank_evaluation import Ranker
from data_reader import read_deriv, FeatureExtractor
from utils import make_vocabulary_matrix
import argparse
import toml
import numpy as np
import os
import csv
from Analysis.results_evaluation import average_results_derinf

def save_baseform_vectors(baseforms, extractor, out_path):
    print( len(baseforms), baseforms[:30])
    predictions = extractor.get_array_embeddings(baseforms)
    print(len(predictions))
    np.save(file=out_path, arr=np.array(predictions), allow_pickle=True)


def main(config):
    relations, base_form, target_words = read_deriv(config['test_path'])
    feature_extractor = FeatureExtractor(config['embeddings'], config['embedding_dim'])
    restr = False
    if config['restricted_vocabulary_matrix'] == "True":
        restr = True
    pred_path = os.path.join(config['out_path'], "predictions.npy")
    save_baseform_vectors(base_form, feature_extractor, pred_path)
    vocabulary_matrix, lab2idx, idx2lab = make_vocabulary_matrix(feature_extractor, config['embedding_dim'], target_words,
                                                                 restricted=restr)
    print("first base forms", base_form[:30])
    print("first target forms", target_words[:30])
    ranker = Ranker(pred_path, target_words, relations, vocabulary_matrix, lab2idx,
                    idx2lab)  # path_predictions, target_words, relations, vocabulary_matrix, lab2idx, idx2lab
    if config['save_detailed']:
        fine_path = os.path.join(config['out_path'], "results_per_word.csv")
        ranker.save_ranks(fine_path)

    acc_at_1 = ranker.precision_at_rank_1
    acc_at_5 = ranker.precision_at_rank_5
    acc_at_50 = ranker.precision_at_rank_50
    acc_at_80 = ranker.precision_at_rank_80
    quartiles = ranker.quartiles
    acc_per_relation = ranker.dict_results_per_relation
    path_out = os.path.join(config['out_path'], "summary.csv")
    path_out2 = os.path.join(config['out_path'], "results_per_relation.csv")
    with open(path_out, 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(("prec_at_1", "prec_at_5","prec_at_50", "prec_at_80", "quartiles"))
        writer.writerow((acc_at_1, acc_at_5,acc_at_50, acc_at_80, quartiles))
    with open(path_out2, 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(("relation", "prec_at_1", "prec_at_5", "prec_at_50", "prec_at_80",  "prediction_sim"))
        for k, v in acc_per_relation.items():
            print("COSINE SIM", v[4])
            writer.writerow((k, v[0], v[1], v[2], v[3], v[4]))
    average_results_derinf(path_out2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    with open(args.config) as cfg_file:
        config = toml.load(cfg_file)
    main(config)
