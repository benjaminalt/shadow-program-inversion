"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""

import torch
import torch.nn as nn


class AutoregressiveModel(nn.Module):
    def __init__(self, static_input_size, recurrent_input_size, lstm_hidden_size):
        """
        :param static_input_size: Size of non-recurrent inputs
        :param recurrent_input_size: Size of recurrent inputs (and outputs)
        :param lstm_hidden_size: Hidden size of LSTMs
        """
        super(AutoregressiveModel, self).__init__()

        self.hidden_net = nn.Sequential(nn.Linear(static_input_size, 32),
                                        nn.SELU(),
                                        nn.Linear(32, 32),
                                        nn.SELU(),
                                        nn.Linear(32, lstm_hidden_size),
                                        nn.SELU())

        self.cell_state_net = nn.Sequential(nn.Linear(static_input_size, 32),
                                            nn.SELU(),
                                            nn.Linear(32, 32),
                                            nn.SELU(),
                                            nn.Linear(32, lstm_hidden_size),
                                            nn.SELU())
        self.lstm_hidden_size = lstm_hidden_size

        self.static_input_size = static_input_size
        self.recurrent_input_size = recurrent_input_size
        self.lstm_input_size = static_input_size + recurrent_input_size

        self.lstm = nn.LSTM(input_size = self.lstm_input_size, hidden_size = lstm_hidden_size, num_layers = 1, batch_first=True)
        self.dropout1 = nn.Dropout()
        self.lstm2 = nn.LSTM(input_size =lstm_hidden_size + self.lstm_input_size, hidden_size = lstm_hidden_size, num_layers = 1, batch_first=True)
        self.dropout2 = nn.Dropout()
        self.lstm3 = nn.LSTM(input_size =lstm_hidden_size * 2, hidden_size = lstm_hidden_size, num_layers = 1, batch_first=True)
        self.dropout3 = nn.Dropout()
        self.lstm4 = nn.LSTM(input_size =lstm_hidden_size * 2, hidden_size = lstm_hidden_size, num_layers = 1, batch_first=True)
        self.dropout4 = nn.Dropout()
        self.linear1 = nn.Linear(lstm_hidden_size * 2, recurrent_input_size)

    def forward(self, static_inputs, recurrent_inputs, hidden=None):
        """
        :param hidden: Initial hidden state. If None, hidden state is initialized from x
        :return: The parameters of a 2D time-variant Gaussian mixture
        """
        (h_1, c_1), (h_2, c_2), (h_3, c_3), (h_4, c_4) = self._init_hidden(static_inputs) if hidden is None else hidden

        x = torch.cat((static_inputs.unsqueeze(1).repeat(1, recurrent_inputs.size(1), 1), recurrent_inputs), -1)
        h1, (h1_n, c1_n) = self.lstm(x, (h_1, c_1))

        h1 = self.dropout1(h1)
        x2 = torch.cat([h1, x], dim=-1)  # skip connection
        h2, (h2_n, c2_n) = self.lstm2(x2, (h_2, c_2))

        h2 = self.dropout2(h2)
        x3 = torch.cat([h1, h2], dim=-1)  # skip connection
        h3, (h3_n, c3_n) = self.lstm3(x3, (h_3, c_3))

        h3 = self.dropout3(h3)
        x4 = torch.cat([h2, h3], dim=-1)  # skip connection
        h4, (h4_n, c4_n) = self.lstm4(x4, (h_4, c_4))

        h4 = self.dropout4(h4)
        h = torch.cat([h3, h4], dim=-1)  # skip connection
        out = self.linear1(h)
        out[:,:,:2] = torch.nn.Sigmoid()(out[:,:,:2])  # eos probability & success probability

        hidden_new = ((h1_n, c1_n), (h2_n, c2_n), (h3_n, c3_n), (h4_n, c4_n))
        return out, hidden_new

    def save(self, filepath: str):
        torch.save({
            "state_dict": self.state_dict(),
            "static_input_size": self.static_input_size,
            "recurrent_input_size": self.recurrent_input_size,
            "lstm_hidden_size": self.lstm_hidden_size
        }, filepath)

    @staticmethod
    def load(filepath: str, device):
        params = torch.load(filepath, map_location=device)
        model = AutoregressiveModel(params["static_input_size"], params["recurrent_input_size"], params["lstm_hidden_size"])
        model.load_state_dict(params["state_dict"])
        return model

    def _init_hidden(self, static_inputs):
        hidden_state = self.hidden_net(static_inputs)  # Shape [batch_size, lstm_hidden_size]
        h_1, h_2, h_3, h_4 = hidden_state.view((1, 1, hidden_state.size(0), hidden_state.size(1))).repeat(4, 1, 1, 1)
        cell_state = self.cell_state_net(static_inputs)  # Shape [batch_size, lstm_hidden_size]
        c_1, c_2, c_3, c_4 = cell_state.view((1, 1, cell_state.size(0), cell_state.size(1))).repeat(4, 1, 1, 1)
        return (h_1, c_1), (h_2, c_2), (h_3, c_3), (h_4, c_4)
