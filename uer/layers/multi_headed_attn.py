import math
import sys
import torch
import torch.nn as nn


class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, heads_num, attention_head_size, dropout, has_bias=True, with_scale = True,weight_squeeze =False):
        super(MultiHeadedAttention, self).__init__()
        self.heads_num = heads_num
        self.weight_squeeze = weight_squeeze
        self.per_head_size = attention_head_size
        self.with_scale = with_scale
        self.inner_hidden_size = heads_num * attention_head_size

        self.linear_layers = nn.ModuleList(
                [nn.Linear(hidden_size, self.inner_hidden_size, bias=has_bias) for _ in range(3)]
            )
        if weight_squeeze:
            self.left_compactor = nn.ModuleList(
                    [nn.Linear(hidden_size, self.inner_hidden_size, bias=False) for _ in range(3)]
                )
            self.right_compactor = nn.ModuleList(
                    [nn.Linear(hidden_size, self.inner_hidden_size, bias=False) for _ in range(3)]
                )
            self.final_linear_left_compactor = nn.Linear(hidden_size, hidden_size, bias=False)
            self.final_linear_right_compactor = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(self.inner_hidden_size, hidden_size, bias=has_bias)

    def forward(self, key, value, query, mask, position_bias=None, has_residual_attention=False, prev_attn=None):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]
            position_bias: [1 x heads_num x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """

        if self.weight_squeeze:
            query, key, value = [l(x) for l, x in zip(self.left_compactor, (query, key, value))]
        batch_size, seq_length, _ = query.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                   contiguous(). \
                   view(batch_size, seq_length, heads_num, per_head_size). \
                   transpose(1, 2)

        def unshape(x):
            return x. \
                   transpose(1, 2). \
                   contiguous(). \
                   view(batch_size, seq_length, self.inner_hidden_size)


        query, key, value = [l(x). \
                             view(batch_size, -1, heads_num, per_head_size). \
                             transpose(1, 2) \
                             for l, x in zip(self.linear_layers, (query, key, value))
                            ]
        
        if self.weight_squeeze:
            query, key, value = [l(unshape(x)). \
                                 view(batch_size, -1, heads_num, per_head_size). \
                                 transpose(1, 2) \
                                 for l, x in zip(self.right_compactor, (query, key, value))
                                ]
        scores = torch.matmul(query, key.transpose(-2, -1))
        if position_bias is not None:
            scores = scores + position_bias
        if self.with_scale:
            scores = scores / math.sqrt(float(per_head_size))
        scores = scores + mask.type_as(scores)
 #       scores[:,2,:,:] = -10000
 #       scores[:,3,:,:] = -10000

        prev_attn_out = None
        if has_residual_attention:
            if prev_attn is not None:
                scores += prev_attn
            prev_attn_out = scores
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))

        if self.weight_squeeze:
            output = self.final_linear_left_compactor(output)
            output = self.final_linear_right_compactor(self.final_linear(output))
        else:
            output = self.final_linear(output)
        return output, prev_attn_out
