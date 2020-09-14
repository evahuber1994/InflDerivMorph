import argparse
import toml
import torch
from model import RelationFeedForward
from utils import make_vocabulary_matrix, shuffle_lists
from data_reader import create_label_encoder, FeatureExtractor, RelationsDataLoader, read_deriv
import os
from torch import optim
import torch.nn as nn
import numpy as np
from rank_evaluation import Ranker
import csv
def train(train_loader, val_loader, model, model_path, nr_epochs, patience):
    optimizer = optim.Adam(model.parameters())  # or make an if statement for choosing an optimizer
    current_patience = patience
    # train_loss = 0.0
    best_cos = 0.0
    best_model = None
    loss = nn.MSELoss()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    total_cos_similarities = []
    for epoch in range(1, nr_epochs + 1):
        print('epoch {}'.format(epoch))

        model.train()
        val_cos_sim = []
        for batch in train_loader:
            out = model(batch)
            out_loss = loss(out, batch['w2'])
            out_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        model.eval()
        for batch in val_loader:
            out = model(batch)
            val_loss = loss(out, batch['w2'])
            # valid_losses.append(loss.item())
            cosines = cos(out, batch['w2'])
            val_cos_sim.append(sum(cosines) / len(cosines))
        mean_cos = sum(val_cos_sim) / len(val_cos_sim)
        total_cos_similarities.append(mean_cos)

        if mean_cos > best_cos:
            current_patience = patience
            best_model = model
            best_cos = mean_cos
        else:
            current_patience -= 1
        if current_patience < 1:
            try:
                save_model(best_model, model_path)
                print("stopped after epoch {}, cosine similarity {}".format(epoch, best_cos))
                break
            except:
                print("could not save model, model is none")
                break

    if current_patience > 0:
        print("finishes after all epochs, cosine similarity {}".format(best_cos))
        save_model(best_model, model_path)

def predict(model_path, data_loader):
    model = torch.load(model_path)
    print("MODEL", model_path)
    model.eval()
    predictions = []
    true_word_forms = []

    for batch in data_loader:
        print(batch['w1_form'], batch['w2_form'])
        out = model(batch['w1'])
        for pred in out:
            predictions.append(pred.detach().numpy())
        for wf in batch['w2_form']:
            true_word_forms.append(wf)

    return predictions, true_word_forms
def save_model(model, path):
    torch.save(model, path)

def save_predictions(path, predictions):
    np.save(file=path, arr=np.array(predictions), allow_pickle=True)

def save_embeddings(path, model_path, encoder):
    model = torch.load(model_path)
    with open(path, 'w') as file:
        for i, emb in enumerate(model.relation_embeddings):
            file.write("{} \t {}".format(encoder.transform(i), emb))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('all_models', action='store_false')
    args = parser.parse_args()
    with open(args.config) as cfg_file:
        config = toml.load(cfg_file)

    feature_extractor = FeatureExtractor(config['embeddings'], embedding_dim=config['embedding_dim'])
    print("loaded embeddings")
    vocabulary_matrix, lab2idx, idx2lab = make_vocabulary_matrix(feature_extractor, config['embedding_dim'])
    print("built vocabulary matrix")

    train_path = config['train_path']
    test_path = config['test_path']
    val_path = config['val_path']

    emb_path = os.path.join(config['out_path'], "embeddings")
    pred_path = os.path.join(config['out_path'], "predictions")


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
    model = RelationFeedForward(emb_dim=config['embedding_dim'], emb_dim_rels=config['rel_embedding_dim'],
                                hidden_dim=config['hidden_dim'], relation_nr=len(all_relations),
                                dropout_rate=config['dropout_rate'], non_lin=config['non_linearity'], function=config['non_linearity_function'],
                                layers=config['nr_layers'])

    train(train_l, val_l, model, config['model_path'], config['nr_epochs'], config['patience'])
    predictions, target_word_forms = predict(config['model_path'], test_l)
    save_predictions(pred_path, predictions)
    save_embeddings(emb_path, config['model_path'], encoder)
    ranker = Ranker(path_predictions=pred_path, target_words=target_word_forms, vocabulary_matrix=vocabulary_matrix,
                    lab2idx=lab2idx, idx2lab=idx2lab)
    if config['save_detailed']:
        fine_path = os.path.join(config['out_path'], "_results_per_relation.csv")
        ranker.save_metrics_per_relation(fine_path)

    acc_at_1 = ranker.precision_at_rank_1
    acc_at_5 = ranker.precision_at_rank_5
    quartiles = ranker.quartiles
    acc, f1 = ranker.performance_metrics()
    path_out = os.path.join(config['out_path'], "_summary.csv")
    with open(path_out, 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(("acc_at_1", "acc_at_5", "quartiles", "acc", "f1"))
        writer.writerow((acc_at_1, acc_at_5, quartiles, acc, f1))

    average_rank = sum(ranker.ranks) / len(ranker.ranks)
    average_rr = sum(ranker.reciprank) / len(ranker.reciprank)
    average_sim = sum(ranker.preds_sims) / len(ranker.preds_sims)
    print("avg rank: {}, avg rr: {}, avg sim: {}".format(average_rank, average_rr, average_sim))


if __name__ == '__main__':
    main()
