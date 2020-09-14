from data_reader import SimpleDataLoader, read_deriv, FeatureExtractor, create_label_encoder
from model import BasicFeedForward, RelationFeedForward
from torch import optim
import torch.nn as nn
import torch
import numpy as np
import os
import argparse
import toml
import csv
from utils import make_vocabulary_matrix, shuffle_lists, cosine_distance_loss
from rank_evaluation import Ranker
from sklearn.metrics.pairwise import cosine_similarity






def train(train_loader, val_loader, model, model_path, nr_epochs, patience, loss_type):
    optimizer = optim.Adam(model.parameters())  # or make an if statement for choosing an optimizer
    current_patience = patience
    # train_loss = 0.0
    best_cos = 0.0
    best_model = None
    if loss_type == 'cosine_distance':
        loss = cosine_distance_loss()
    elif loss_type == 'mse':
        loss = nn.MSELoss()
    else:
        raise Exception("invalid loss type given, has to be \'cosine_distance\' or \'mse\'")
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    total_cos_similarities = []

    for epoch in range(1, nr_epochs + 1):
        print('epoch {}'.format(epoch))
        model.train()
        val_cos_sim = []
        # for word1, word2, labels in train_loader:
        for batch in train_loader:
            out = model(batch['w1'])
            out_loss = loss(out, batch['w2'])
            out_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # train_losses.append(loss.item())
        # validation loop over validation batches
        model.eval()
        for batch in val_loader:
            out = model(batch['w1'])
            val_loss = loss(out, batch['w2'])
            #valid_losses.append(loss.item())
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


def save_model(model, path):
    torch.save(model, path)


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


def save_predictions(path, predictions):
    np.save(file=path, arr=np.array(predictions), allow_pickle=True)


def evaluate():
    pass





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
    dict_results = dict()
    for subdir in os.listdir(config['data_path']):
        train_path = ''
        val_path = ''
        test_path = ''
        pred_path = ''
        subdir = os.path.join(config['data_path'], subdir)
        files = os.listdir(subdir)
        for f in files:
            print("file",f)
            if f.endswith('train.csv'):
                train_path = os.path.join(subdir, f)
                file = f.strip('_train.csv')
                name_pred = f.strip('train.csv') + "predictions.npy"
                name_res = f.strip('train.csv') + "results.csv"
                pred_path = os.path.join(config['out_path'], name_pred)
                res_path = os.path.join(config['out_path'], name_res)
            elif f.endswith('val.csv'):
                val_path = os.path.join(subdir, f)
            elif f.endswith('test.csv'):
                test_path = os.path.join(subdir, f)
            else:
                raise Exception("wrong file path in directory: {}".format(f))
        _, word1_train, word2_train = read_deriv(train_path)
        _, word1_val, word2_val = read_deriv(val_path)
        _, word1_test, word2_test = read_deriv(test_path)

        data_train = SimpleDataLoader(feature_extractor, word1_train, word2_train)
        data_val = SimpleDataLoader(feature_extractor, word1_val, word2_val)
        data_test = SimpleDataLoader(feature_extractor, word1_test, word2_test)


        train_l = torch.utils.data.DataLoader(data_train, batch_size=config['batch_size'])
        val_l = torch.utils.data.DataLoader(data_val, batch_size=config['batch_size'])
        test_l = torch.utils.data.DataLoader(data_test, batch_size=config['batch_size'])
        # input_dim, hidden_dim, label_nr, dropout_rate=0, non_lin=True, function='sigmoid', layers=1

        model = BasicFeedForward(input_dim=config['embedding_dim'], hidden_dim=config['hidden_dim'],
                                      label_nr=config['embedding_dim'], dropout_rate=config['dropout'],
                                      non_lin=config['non_linearity'], function=config['non_linearity_function'], layers=config['nr_layers'])


        train(train_l, val_l, model, config['model_path'], config['nr_epochs'], config['patience'], config['loss'])

        predictions, target_word_forms = predict(config['model_path'], test_l)

        save_predictions(pred_path, predictions)
        ranker = Ranker(path_predictions=pred_path, target_words=target_word_forms, vocabulary_matrix=vocabulary_matrix,
                        lab2idx=lab2idx, idx2lab=idx2lab)
        if config['save_detailed']:
            fine_grained = config['out_path'] + str(file)
            ranker.save_metrics_per_relation(fine_grained)


        #average_rank = sum(ranker.ranks)/len(ranker.ranks)
        #average_rr = sum(ranker.reciprank)/len(ranker.reciprank)
        #average_sim = sum(ranker.preds_sims)/len(ranker.preds_sims)
        acc, f1 = ranker.performance_metrics()
        dict_results[str(file)] = (ranker.precision_at_rank_1,ranker.precision_at_rank_5, acc, f1)

    out_summary_path = os.path.join(config['out_path'], "results_all.csv")
    with open(out_summary_path, 'w') as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(("relation", "precision_at_rank1", "precision_at_rank5", "accuracy", "f1"))
        for k,v in dict_results.items():
            writer.writerow((k,v[0], v[1], v[2], v[3]))
    """
    out_summary_path = os.path.join(config['out_path'], "summary.csv")

    
    avg_r = []
    avg_rr = []
    avg_sim = []
    with open(out_summary_path, 'w') as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(("relation", "avg_rank", "avg_reciprank", "avg_sim"))
        for k,v in dict_results.items():
            writer.writerow((k, v[0], v[1], v[2]))
            avg_r.append(v[0])
            avg_rr.append(v[1])
            avg_sim.append(v[2])
    print("average rank of config: {}, average reciprank of config: {}, average sim of config: {}".format(float(sum(avg_r)/len(avg_r)), float(sum(avg_rr)/len(avg_rr)), float(sum(avg_sim)/len(avg_sim))))
    """
        ###save average of all models

    # if file endswith train.csv if file endswith

    # path_train = 'dNA22_train.csv'
    # path_val = 'dNA22_val.csv'
    # path_test = 'dNA22_test.csv'
    # embs = 'word2vec-mincount-30-dims-100-ctx-10-ns-5.w2v'
    # embedding_dim = 200
    # epochs = 100
    # patience = 10
    # batch_size = 12
    #
    # _, word1_train, word2_train = read_deriv(path_train)
    # _, word1_val, word2_val = read_deriv(path_val)
    # _, word1_test, word2_test = read_deriv(path_test)
    # data_train = SimpleDataLoader(feature_extractor, word1_train, word2_train)
    # data_val = SimpleDataLoader(feature_extractor, word1_val, word2_val)
    # data_test = SimpleDataLoader(feature_extractor, word1_test, word2_test)
    # train_l = torch.utils.data.DataLoader(data_train, batch_size=batch_size)
    # val_l = torch.utils.data.DataLoader(data_val, batch_size=batch_size)
    # test_l = torch.utils.data.DataLoader(data_test, batch_size=batch_size)
    # # input_dim, hidden_dim, label_nr, dropout_rate=0, non_lin=True, function='sigmoid', layers=1
    # model1 = BasicFeedForward(embedding_dim, embedding_dim, embedding_dim, non_lin=True)
    # model_path = 'output/model/model1'
    # # train_loader, val_loader, model, model_path, nr_epochs, patience
    # train(train_l, val_l, model1, model_path, epochs, patience)
    #
    # predictions, target_word_forms = predict(model_path, test_l)
    # path_pred = 'output/preds.npy'
    # save_predictions(path_pred, predictions)
    # # save predictions
    # path_results = 'output/results.txt'
    # # path_pred = os.path.join(path_pred, '.npy')
    # ranker = Ranker(path_predictions=path_pred, target_words=target_word_forms, vocabulary_matrix=vocabulary_matrix,
    #                 lab2idx=lab2idx, idx2lab=idx2lab, path_results=path_results)


if __name__ == "__main__":
    main()
