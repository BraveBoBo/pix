import torch
import numpy as np
from torch import nn
import tools


class Mlp(nn.Module):
    def __init__(self, input_number, hidden_numbers, is_training=True):  # hidden_numbers 是一个元组
        super(Mlp, self).__init__()
        self.inputsize = input_number
        self.hidden_numbers = hidden_numbers
        self.is_training = is_training
        self.linear1 = nn.Linear(input_number, hidden_numbers[0])
        self.linear2 = nn.Linear(hidden_numbers[0], hidden_numbers[1])
        self.linear3 = nn.Linear(hidden_numbers[1], hidden_numbers[2])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        if self.is_training:
            x = tools.dropout_layers(x, 0.5)
        x = self.relu(self.linear2(x))
        if self.is_training:
            x = tools.dropout_layers(x, 0.5)
        return self.linear3(x)


# net = Mlp(3, (6, 4, 3))
# print(net)
