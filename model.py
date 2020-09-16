import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch
import numpy as np
class BasicFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, label_nr, dropout_rate=0, non_lin=True, function='sigmoid', layers=1):
        super(BasicFeedForward, self).__init__()
        self._label_nr = label_nr
        self._dropout_rate = dropout_rate

        self._non_lin = non_lin
        self._layers = layers

        self._hidden_layer = OrderedDict()
        for i in range(1, layers + 1):
            if i > 1:
                self._hidden_layer[str(i) + "LF"] = nn.Linear(hidden_dim, hidden_dim)
            else:
                self._hidden_layer[str(i) + "LF"] = nn.Linear(input_dim, hidden_dim)
            if non_lin:
                self._hidden_layer[str(i) + "NL"] = nn.Sigmoid()
            self._hidden_layer[str(i) + "D"] = nn.Dropout(p=self._dropout_rate)

        self._hidden_layer['output'] = nn.Linear(hidden_dim, label_nr)
        self._model = nn.Sequential(self._hidden_layer)  # second argument is output layer

    def forward(self, batch):
        return self._model(batch['w1'])

class RelationFeedForward(nn.Module):
    def __init__(self, emb_dim, emb_dim_rels, hidden_dim, relation_nr, dropout_rate=0, non_lin=True, function='sigmoid', layers=1):
        super(RelationFeedForward, self).__init__()
        self._emb_dim = emb_dim
        self._emb_dim_rels = emb_dim_rels
        self._dropout_rate = dropout_rate
        self._non_lin = non_lin
        self._layers = layers
        self._hidden_layer = OrderedDict()
        self._relation_embeddings =  nn.Embedding(relation_nr, emb_dim_rels)
        if function == 'sigmoid':
            non_lin_function = nn.Sigmoid()
        elif function == 'relu':
            non_lin_function = nn.ReLU()
        elif function == 'tanh':
            non_lin_function = nn.Tanh()
        else:
            raise Exception("invalid non-linearity function")

        for i in range(1, layers + 1):
            if i > 1:
                self._hidden_layer[str(i) + "LF"] = nn.Linear(hidden_dim, hidden_dim)
            else:
                self._hidden_layer[str(i) + "LF"] = nn.Linear(emb_dim + emb_dim_rels, hidden_dim)
            if self._non_lin == True:
                self._hidden_layer[str(i) + "NL"] = non_lin_function
            self._hidden_layer[str(i) + "D"] = nn.Dropout(p=self._dropout_rate)

        self._hidden_layer['output'] = nn.Linear(hidden_dim, emb_dim)
        self._model = nn.Sequential(self._hidden_layer)  # second argument is output layer

    def forward(self,batch):
        concat_vector = concatenate(batch['w1'].to(batch['device']), self._relation_embeddings(batch['rel'].squeeze().to(batch['device'])), 1)
        return self.model(concat_vector)


    @property
    def relation_embeddings(self):
        return self._relation_embeddings

    @property
    def model(self):
        return self._model

def concatenate(v1, v2, axis):
    return torch.cat((v1,v2), axis)
def main():
    #elf, emb_dim, emb_dim_rels, hidden_dim, relation_nr, dropout_rate=0, non_lin=True, function='sigmoid', layers=1

    tensor_1 = nn.Parameter(torch.from_numpy(np.array([1,2,3,4,5], dtype='double')))
    model = BasicFeedForward(5, 5, 3, non_lin = False)
    model2 = BasicFeedForward(5, 5, 3, non_lin =True)
    out = model(tensor_1.float())
    out2 = model2(tensor_1.float())
    print(out)
    print(out2)

if __name__ == "__main__":
    main()
