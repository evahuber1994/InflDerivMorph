from data_reader import SimpleDataLoader, read_deriv, FeatureExtractor
from model import BasicFeedForward
from torch import optim
import torch.nn as nn
import torch
from sklearn.metrics.pairwise import cosine_similarity


def train(train_loader, val_loader, model, model_path, nr_epochs, patience):
    optimizer = optim.Adam(model.parameters())  # or make an if statement for choosing an optimizer
    current_patience = patience
    best_epoch = 0
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
            # valid_losses.append(loss.item())
            cosines = cos(out, batch['w2'])
            val_cos_sim.append(sum(cosines)/len(cosines))

        mean_cos = sum(val_cos_sim) / len(val_cos_sim)
        total_cos_similarities.append(mean_cos)

        if mean_cos > best_cos:
            current_patience = patience
            best_epoch = epoch
            best_model = model
            best_cos = mean_cos
        else:
            current_patience -= 1
        if current_patience < 1:
            try:
                save_model(best_model, model_path + 'epoch' + str(best_epoch))
                print("stopped after epoch {}, cosine similarity {}".format(epoch, best_cos))
                break
            except:
                print("could not save model, model is none")
                break

    if current_patience > 0:
        print("finishes after all epochs, cosine similarity {}".format(best_cos))
        save_model(best_model, model_path + 'epoch' + str(best_epoch))


def save_model(model, path):
    torch.save(model, path)


def predict(model_path, data_loader):
    model = torch.load(model_path)
    model.eval()
    predictions = []
    true_word_forms = []

    for batch in data_loader:
        out = model(batch['w1'])
        for pred in out:
            predictions.append(pred.detach().numpy())
        for wf in batch['w2_form']:
            true_word_forms.append(wf)

    return predictions, true_word_forms


    # sims = []
    # print("LENGHTS", len(predictions), len(true_word_forms))
    # for p, t in zip(predictions, true_word_forms):
    #     print("T", t)
    #     emb2 = extr.get_embedding(t)
    #     print(p.shape, emb2.shape)
    #     sims.append(cosine_similarity(p.reshape(1, -1),emb2.reshape(1, -1)))
    # print(sims)
def save_predictions():
    pass

def evaluate():
    pass


def main():
    #emb = 'de_core_news_lg'
    path_train = 'data/files_per_relation/splits/dAA01_train.csv'
    path_val = 'data/files_per_relation/splits/dAA01_val.csv'
    path_test = 'data/files_per_relation/splits/dAA01_test.csv'
    embs = 'embeddings/german-skipgram-mincount-30-ctx-10-dims-300.fifu'
    feature_extractor = FeatureExtractor(embs)
    _, word1_train, word2_train = read_deriv(path_train)
    _, word1_val, word2_val = read_deriv(path_val)
    _, word1_test, word2_test = read_deriv(path_test)
    data_train = SimpleDataLoader(feature_extractor, word1_train, word2_train)
    data_val = SimpleDataLoader(feature_extractor, word1_val, word2_val)
    data_test = SimpleDataLoader(feature_extractor, word1_test, word2_test)
    train_l = torch.utils.data.DataLoader(data_train, batch_size=12)
    val_l = torch.utils.data.DataLoader(data_val, batch_size=12)
    test_l = torch.utils.data.DataLoader(data_test, batch_size=12)
    # input_dim, hidden_dim, label_nr, dropout_rate=0, non_lin=True, function='sigmoid', layers=1
    model1 = BasicFeedForward(300, 300, 300, non_lin=True)
    # train_loader, val_loader, model, model_path, nr_epochs, patience
    #train(train_l, val_l, model1, 'data/trained_models/', 100, 10)
    path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data/trained_models/epoch100'
    predict(path, test_l)


if __name__ == "__main__":
    main()
