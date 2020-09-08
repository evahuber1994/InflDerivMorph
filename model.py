import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class BasicFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, label_nr, dropout_rate=0, non_lin=True, function='sigmoid', layers=1):
        super(BasicFeedForward, self).__init__()
        self._label_nr = label_nr
        self._dropout_rate = dropout_rate
        self._non_lin = non_lin
        self._layers = layers

        self._hidden_layer = OrderedDict()
        for i in range(1, layers + 1):

            self._hidden_layer[str(i) + "LF"] = nn.Linear(input_dim, hidden_dim)
            if non_lin:
                self._hidden_layer[str(i) + "NL"] = nn.Sigmoid()
            self._hidden_layer[str(i) + "D"] = nn.Dropout(p=self._dropout_rate)

        self._hidden_layer['output'] = nn.Linear(hidden_dim, label_nr)
        self._model = nn.Sequential(self._hidden_layer)  # second argument is output layer

    def forward(self, batch):
        return self._model(batch)


def main():
    model = BasicFeedForward(10, 10, 3, layers=4)



if __name__ == "__main__":
    main()
