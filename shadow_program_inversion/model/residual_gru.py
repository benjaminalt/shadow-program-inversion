"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""

import torch
from torch import nn


class ResidualGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout_p):
        super(ResidualGRU, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        self.grus = nn.ModuleList([nn.GRU(2 * hidden_size, hidden_size,
                                          batch_first=True) for _ in range(self.num_layers)])
        self.output_layer = nn.Linear(self.hidden_size, output_size)
        all_params = sum(p.numel() for p in self.parameters())
        print("ResidualGRU, total number of parameters: {}".format(all_params))

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward(self, x, hidden):
        x = nn.SELU()(self.input_layer(x))
        x_prev = x
        for i in range(len(self.grus)):
            gru_input = torch.cat((x_prev, x), dim=-1)
            x_prev = x
            x, hidden = self.grus[i](gru_input, hidden)
            x = nn.SELU()(x)
            x = nn.Dropout(self.dropout_p)(x)
        x = self.output_layer(x)
        x[:, :, :2] = nn.Sigmoid()(x[:, :, :2])
        return x

    def save(self, filepath: str):
        torch.save({
            "state_dict": self.state_dict(),
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout_p": self.dropout_p
        }, filepath)

    @staticmethod
    def load(filepath: str, device):
        params = torch.load(filepath, map_location=device)
        model = ResidualGRU(params["input_size"], params["output_size"], params["hidden_size"], params["num_layers"],
                            params["dropout_p"])
        model.load_state_dict(params["state_dict"])
        return model
