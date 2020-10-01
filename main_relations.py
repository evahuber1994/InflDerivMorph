import argparse
import toml
import torch
from model import RelationFeedForward
from utils import make_vocabulary_matrix, shuffle_lists, cosine_distance_loss
from data_reader import create_label_encoder, FeatureExtractor, RelationsDataLoader, read_deriv
import os
from torch import optim
import torch.nn as nn
import numpy as np
from rank_evaluation import Ranker
import csv


def train(train_loader, val_loader, config, length_relations, device):
    # or make an if statement for choosing an optimizer

    # train_loss = 0.0
    tolerance = 1e-5
    if config['non_linearity'] == "True":
        non_lin = True
    else:
        non_lin = False
    model = RelationFeedForward(emb_dim=config['embedding_dim'], emb_dim_rels=config['rel_embedding_dim'],
                                hidden_dim=config['hidden_dim'], relation_nr=length_relations,
                                dropout_rate=config['dropout_rate'], non_lin=non_lin,
                                function=config['non_linearity_function'],
                                layers=config['nr_layers'])
    print("model", model)
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    current_patience = config['patience']
    best_cos = 0.0
    lowest_loss = float('inf')
    best_model = None
    loss_type = config['loss_type']
    assert loss_type == 'cosine_distance' or loss_type == 'mse', "loss type has to be either \' cosine distance\' or \'mse\'"

    if loss_type == 'mse':
        loss = nn.MSELoss()
    tolerance = 1e-5
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    total_cos_similarities = []
    valid_losses = []
    for epoch in range(1, config['nr_epochs'] + 1):
        print('epoch {}'.format(epoch))

        model.train()
        val_cos_sim = []
        for batch in train_loader:
            batch['device'] = device
            out = model(batch)
            out = out.to('cpu')
            if loss_type == 'mse':
                out_loss = loss(out, batch['w2'])
            else:
                out_loss = cosine_distance_loss(out, batch['w2'])
            out_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        model.eval()
        for batch in val_loader:
            batch['device'] = device
            out = model(batch)
            out = out.to('cpu')
            if loss_type == 'mse':
                val_loss = loss(out, batch['w2'])
            else:
                val_loss = cosine_distance_loss(out, batch['w2'])
            valid_losses.append(val_loss.item())
            cosines = cos(out, batch['w2'])
            val_cos_sim.append(sum(cosines) / len(cosines))
        mean_cos = sum(val_cos_sim) / len(val_cos_sim)
        mean_loss = np.average(valid_losses)
        print("epoch: {}, mean cosine similarity: {}, mean validation loss: {}".format(epoch, mean_cos, mean_loss))
        total_cos_similarities.append(mean_cos)
        if config['early_stopping_criterion'] == 'cosine_similarity':
            if mean_cos > best_cos - tolerance: #valid_f1 > best_f1 - tolerance:
                current_patience = config['patience']
                best_model = model
                best_cos = mean_cos
                save_model(best_model, config['model_path'])
            else:
                current_patience -= 1
        else:
            if lowest_loss - mean_loss > tolerance:
                lowest_loss = mean_loss
                best_model = model
                current_patience = config['patience']
                save_model(best_model, config['model_path'])
            else:
                current_patience -= 1
        if current_patience < 1:
            print("finishes training after epoch: {}".format(epoch))
            break


def predict(model_path, data_loader, device):
    model = torch.load(model_path)
    print("MODEL", model_path)
    model.to(device)
    model.eval()
    predictions = []
    true_word_forms = []
    relations = []

    for batch in data_loader:
        batch['device'] = device
        out = model(batch)
        out = out.to('cpu')
        for pred in out:
            predictions.append(pred.detach().numpy())
        for wf in batch['w2_form']:
            true_word_forms.append(wf)
        for rel in batch['rel_form']:
            relations.append(rel)

    return predictions, true_word_forms, relations


def save_model(model, path):
    torch.save(model, path)


def save_predictions(path, predictions):
    np.save(file=path, arr=np.array(predictions), allow_pickle=True)



def save_embeddings(path, model_path, encoder, relations):
    model = torch.load(model_path)
    model.to('cpu')
    weights = model.relation_embeddings.weight
    print("weights type {}".format(type(weights)))
    print("weights shape {}".format(weights.shape))
    with open(path, 'wt') as fp:
        fmt = "{}\t{}" + (weights.shape[1] - 1) * " {}"
        print(str(weights.shape[1] - 1))
        for w in relations:
            e = weights[encoder.transform([w])]
            e = e.detach().numpy()
            e = e.tolist()
            e = e[0]
            print(fmt.format(w, *e), file=fp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('all_models', action='store_false')
    args = parser.parse_args()
    with open(args.config) as cfg_file:
        config = toml.load(cfg_file)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(config['device'])
    feature_extractor = FeatureExtractor(config['embeddings'], embedding_dim=config['embedding_dim'])
    print("loaded embeddings")


    train_path = config['train_path']
    test_path = config['test_path']
    val_path = config['val_path']

    emb_path = os.path.join(config['out_path'], "embeddings.txt")
    pred_path = os.path.join(config['out_path'], "predictions.npy")

    relation_train, word1_train, word2_train = read_deriv(train_path, shuffle=True)
    relation_val, word1_val, word2_val = read_deriv(val_path, shuffle=True)
    relation_test, word1_test, word2_test = read_deriv(test_path, shuffle=True)
    all_relations = set(relation_train + relation_val + relation_test)

    encoder = create_label_encoder(list(all_relations))
    data_train = RelationsDataLoader(feature_extractor, word1_train, word2_train, relation_train, encoder)
    data_val = RelationsDataLoader(feature_extractor, word1_val, word2_val, relation_val, encoder)
    data_test = RelationsDataLoader(feature_extractor, word1_test, word2_test, relation_test, encoder)

    train_l = torch.utils.data.DataLoader(data_train, batch_size=config['batch_size'])
    val_l = torch.utils.data.DataLoader(data_val, batch_size=config['batch_size'])
    test_l = torch.utils.data.DataLoader(data_test, batch_size=config['batch_size'])

    print("number of relations:", len(all_relations))
    train(train_l, val_l, config, len(all_relations), device)
    predictions, target_word_forms, relations = predict(config['model_path'], test_l, device)
    save_predictions(pred_path, predictions)
    save_embeddings(emb_path, config['model_path'], encoder, all_relations)
    print("finished writing embeddings")
    restr = False
    if config['restricted_vocabulary_matrix'] == "True":
        restr = True

    vocabulary_matrix, lab2idx, idx2lab = make_vocabulary_matrix(feature_extractor, config['embedding_dim'], target_word_forms, restricted=restr)
    print("built vocabulary matrix; shape is {}".format(vocabulary_matrix.shape))
    ranker = Ranker(path_predictions=pred_path, target_words=target_word_forms, relations=relations,
                    vocabulary_matrix=vocabulary_matrix,
                    lab2idx=lab2idx, idx2lab=idx2lab)
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

    # print("prediction sim", type(ranker.prediction_similarities), ranker.prediction_similarities)
    # average_rank = sum(ranker.ranks) / len(ranker.ranks)
    # average_rr = sum(ranker.reciprank) / len(ranker.reciprank)
    # average_sim = sum(ranker.prediction_similarities) / len(ranker.prediction_similarities)
    # print("avg rank: {}, avg rr: {}, avg sim: {}".format(average_rank, average_rr, average_sim))


if __name__ == '__main__':
    main()
