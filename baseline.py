from rank_evaluation import Ranker
from data_reader import read_deriv, FeatureExtractor
from utils import make_vocabulary_matrix
import argparse
import toml
import numpy as np
import os
import csv


def save_baseform_vectors(baseforms, extractor, out_path):
    predictions = extractor.get_array_embeddings(baseforms)
    np.save(file=out_path, arr=np.array(predictions), allow_pickle=True)


def main(config):
    relations, base_form, target_words = read_deriv(config['test_path'])
    feature_extractor = FeatureExtractor(config['embeddings'], config['embedding_dim'])
    restr = False
    if config['restricted_vocabulary_matrix'] == "True":
        restr = True
    pred_path = os.path.join(config['out_path'], "predictions.npy")
    save_baseform_vectors(base_form, feature_extractor, pred_path)
    vocabulary_matrix, lab2idx, idx2lab = make_vocabulary_matrix(feature_extractor, config['embedding_dim'], base_form,
                                                                 restricted=restr)

    ranker = Ranker(pred_path, target_words, relations, vocabulary_matrix, lab2idx,
                    idx2lab)  # path_predictions, target_words, relations, vocabulary_matrix, lab2idx, idx2lab
    if config['save_detailed']:
        fine_path = os.path.join(config['out_path'], "results_per_word.csv")
        ranker.save_ranks(fine_path)

    acc_at_1 = ranker.precision_at_rank_1
    acc_at_5 = ranker.precision_at_rank_5
    quartiles = ranker.quartiles
    acc_per_relation = ranker.dict_results_per_relation
    path_out = os.path.join(config['out_path'], "summary.csv")
    path_out2 = os.path.join(config['out_path'], "results_per_relation.csv")
    with open(path_out, 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(("acc_at_1", "acc_at_5", "quartiles"))
        writer.writerow((acc_at_1, acc_at_5, quartiles))
    with open(path_out2, 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(("relation", "acc_at_1", "acc_at_5"))
        for k, v in acc_per_relation.items():
            writer.writerow((k, v[1], v[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    with open(args.config) as cfg_file:
        config = toml.load(cfg_file)
    main(config)
