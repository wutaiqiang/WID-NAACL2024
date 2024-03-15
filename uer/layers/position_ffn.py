import torch.nn as nn
import sys
from uer.utils import *

class PositionwiseFeedForward(nn.Module):
    """ Feed Forward Layer. """
    def __init__(self, hidden_size, feedforward_size, hidden_act, has_bias=True,weight_squeeze=False):
        super(PositionwiseFeedForward, self).__init__()
        self.weight_squeeze = weight_squeeze
        if weight_squeeze:
            self.linear_1_left_compactor = nn.Linear(hidden_size, hidden_size, bias=False)
            self.linear_1_right_compactor = nn.Linear(feedforward_size, feedforward_size, bias=False)
            self.linear_2_left_compactor = nn.Linear(feedforward_size, feedforward_size, bias=False)
            self.linear_2_right_compactor = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_1 = nn.Linear(hidden_size, feedforward_size, bias=has_bias)
        self.linear_2 = nn.Linear(feedforward_size, hidden_size, bias=has_bias)
        self.act = str2act[hidden_act]

    def forward(self, x):
        if self.weight_squeeze:
            x = self.linear_1_left_compactor(x)
            inter = self.act(self.linear_1_right_compactor(self.linear_1(x)))
            inter = self.linear_2_left_compactor(inter)
            output = self.linear_2_right_compactor(self.linear_2(inter))
        else:
            inter = self.act(self.linear_1(x))
            output = self.linear_2(inter)
        return output


class GatedFeedForward(nn.Module):
    """ Feed Forward Layer with Gated Linear Unit.
        https://arxiv.org/abs/2002.05202
    """
    def __init__(self, hidden_size, feedforward_size, hidden_act, has_bias=True):
        super(GatedFeedForward, self).__init__()
        self.linear_gate = nn.Linear(hidden_size, feedforward_size, bias=has_bias)
        self.linear_1 = nn.Linear(hidden_size, feedforward_size, bias=has_bias)
        self.linear_2 = nn.Linear(feedforward_size, hidden_size, bias=has_bias)
        self.act = str2act[hidden_act]

    def forward(self, x):
        gate = self.act(self.linear_gate(x))
        inter_linear = self.linear_1(x)
        inter = gate * inter_linear
        output = self.linear_2(inter)

        return output
