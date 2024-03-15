import torch
import torch.nn as nn
import sys

class LayerNorm(nn.Module):
    """
    special for rep
    """
    def __init__(self, args, eps=1e-6):
        hidden_size = args.hidden_size
        super(LayerNorm, self).__init__()

        self.args = args
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, mask_name,mask_dict,current_step):
        if self.args.weight_squeeze is True and current_step>self.args.w_step:
            parms_choice = []
            mask_list = mask_dict[mask_name].tolist()
            for num,i in enumerate(mask_list):
                if i == True:
                    parms_choice.append(num)
            parm_choose = torch.tensor(parms_choice).to(x.device)
            mean = x.index_select(dim=-1, index=parm_choose).mean(-1, keepdim=True)
            std = x.index_select(dim=-1, index=parm_choose).std(-1, keepdim=True)
        else:
            mean = x.mean(-1, keepdim=True)
            std = x.std(-1, keepdim=True)

        hidden_states =  self.gamma * (x-mean) / (std + self.eps)

        return hidden_states + self.beta


class T5LayerNorm(nn.Module):
    """
    Construct a layernorm module in the T5 style No bias and no subtraction of mean.
    """
    def __init__(self, hidden_size, eps=1e-6):

        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states.type_as(self.weight)
