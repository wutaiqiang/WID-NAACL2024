import torch
import collections
import math
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--layers_num", type=int, default=12, help=".")
parser.add_argument("--hidden_size", type=int, default=768, help=".")
parser.add_argument("--compactor_mask_strategy",choices=["head", "dim"], default="dim",
                        help=".")
parser.add_argument("--load_model_path", type=str, required=True, help=".")
parser.add_argument("--output_model_path", type=str, required=True, help=".")
parser.add_argument("--mask_path", type=str, required=True, help=".")
parser.add_argument("--mask_step", type=str, default="10000", help=".")

args = parser.parse_args()

A = torch.load(args.load_model_path)
B = collections.OrderedDict()




def get_dim_mask(mask_position, name):
    parms_choice = []

    if "feed_forward.linear_2_left_compactor" in name:
        for j in range(args.hidden_size*4):
            if j not in mask_position:
                parms_choice.append(j)
    else:
        for j in range(args.hidden_size):
            if j not in mask_position:
                parms_choice.append(j)
    return torch.tensor(parms_choice)


def get_head_mask(mask_position, name):
    parms_choice = []

    if "feed_forward.linear_2_left_compactor" in name:
        for j in range(args["hidden_size"]*4):
            if j not in mask_position:
                parms_choice.append(j)
    elif "self_attn.left_compactor.2.weight" in name or "self_attn.final_linear_left_compactor" in name:
        mask_position.sort()
        new_mask_position = []
        for i in mask_position:
            for k in range(i*64,(i+1)*64):
                new_mask_position.append(k)
        for j in range(args["hidden_size"]):
            if j not in new_mask_position:
                parms_choice.append(j)
    else:
        for j in range(args["hidden_size"]):
            if j not in mask_position:
                parms_choice.append(j)
    return torch.tensor(parms_choice)


# ! For embedding layer
def mul_r(key, mask_r):
    B[key] = A[key].mm(mask_r)

mask_list = dict()
for name,_ in A.items():
    if "weight" in name and "compactor" in name:
        mask_list[name] = None
with open(args.mask_path,"r",encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip('\n').split('\t')
        if line[0] == args.mask_step:
            if args.compactor_mask_strategy == "dim":
                mask_list[line[1].lstrip('module').lstrip('.')] = get_dim_mask(list(map(int,line[2].strip('[').strip(']').split(','))),line[1])
            else:
                mask_list[line[1].lstrip('module').lstrip('.')] = get_head_mask(list(map(int,line[2].strip('[').strip(']').split(','))),line[1])

temp = []
for name in mask_list:
    if ("self_attn.left_compactor.2" in name or "target.mlm_linear_1_left_compactor" in name or "feed_forward.linear_1_left_compactor" in name) and mask_list[name] == None:
        mask_list[name] = mask_list["encoder.transformer.0.self_attn.left_compactor.2.weight"]
for name in mask_list:
    if mask_list[name] == None:
        temp.append(name)
    elif mask_list[name] != None:
        for temp_name in temp:
            mask_list[temp_name] = mask_list[name]
        temp = []
indices = mask_list["embedding.emb_compactor.weight"].to(A["embedding.emb_compactor.weight"].device)

mask_r = A["embedding.emb_compactor.weight"].index_select(dim=0, index=indices)


mul_r("embedding.word_embedding.weight", mask_r=mask_r.T)
mul_r("embedding.position_embedding.weight", mask_r=mask_r.T)
mul_r("embedding.segment_embedding.weight", mask_r=mask_r.T)

B["embedding.layer_norm.gamma"] = A["embedding.layer_norm.gamma"].index_select(dim=0, index=indices)
B["embedding.layer_norm.beta"] = A["embedding.layer_norm.beta"].index_select(dim=0, index=indices)

def mul_l_r(mask_l, key, mask_r):
    tmp = mask_r.mm(A[key])
    B[key] = tmp.mm(mask_l)

def mul_bias(mask_r,key):
     return A[key].unsqueeze(0).mm(mask_r.T).squeeze(0)



for l in range(args.layers_num):  # ! Layer Num
    # kqvl
    for k in range(3):
        indices_r = mask_list["encoder.transformer.{}.self_attn.right_compactor.{}.weight".format(l, k)]
        indices_l = mask_list["encoder.transformer.{}.self_attn.left_compactor.{}.weight".format(l, k)]
        key_r = A["encoder.transformer.{}.self_attn.right_compactor.{}.weight".format(l, k)].index_select(dim=0,index=indices_r.to(indices.device))

        key_l = A["encoder.transformer.{}.self_attn.left_compactor.{}.weight".format(l, k)].index_select(dim=1,index=indices_l.to(indices.device))
        mul_l_r(key_l, "encoder.transformer.{}.self_attn.linear_layers.{}.weight".format(l, k), key_r)

        B["encoder.transformer.{}.self_attn.linear_layers.{}.bias".format(l, k)] = mul_bias(A["encoder.transformer.{}.self_attn.right_compactor.{}.weight".format(l, k)],"encoder.transformer.{}.self_attn.linear_layers.{}.bias".format(l, k)).squeeze(0).index_select(dim=0, index=indices_r.to(indices.device))
    indices_r = mask_list["encoder.transformer.{}.self_attn.final_linear_right_compactor.weight".format(l)]
    indices_l = mask_list["encoder.transformer.{}.self_attn.final_linear_left_compactor.weight".format(l)]
    key_r = A["encoder.transformer.{}.self_attn.final_linear_right_compactor.weight".format(l)].index_select(dim=0,
                                                                                                             index=indices_r.to(indices.device))
    key_l = A["encoder.transformer.{}.self_attn.final_linear_left_compactor.weight".format(l)].index_select(dim=1,
                                                                                                            index=indices_l.to(indices.device))
    mul_l_r(key_l, "encoder.transformer.{}.self_attn.final_linear.weight".format(l), key_r)
    key = "encoder.transformer.{}.self_attn.final_linear.bias".format(l)

    B[key] = mul_bias(A["encoder.transformer.{}.self_attn.final_linear_right_compactor.weight".format(l)],key).squeeze(0).index_select(dim=0, index=indices_r.to(indices.device))
    for kk in ["gamma", "beta"]:
        key = "encoder.transformer.{}.layer_norm_1.{}".format(l,kk)
        B[key] = A[key].index_select(dim=0, index=indices_r.to(indices.device))
    # ffn
    for k in range(1, 3):
        indices_r = mask_list["encoder.transformer.{}.feed_forward.linear_{}_right_compactor.weight".format(l, k)]
        key_r = A["encoder.transformer.{}.feed_forward.linear_{}_right_compactor.weight".format(l, k)].index_select(dim=0, index=indices_r.to(indices.device))
        indices_l = mask_list["encoder.transformer.{}.feed_forward.linear_{}_left_compactor.weight".format(l, k)]
        key_l = A["encoder.transformer.{}.feed_forward.linear_{}_left_compactor.weight".format(l, k)].index_select(dim=1, index=indices_l.to(indices.device))
        mul_l_r(key_l, "encoder.transformer.{}.feed_forward.linear_{}.weight".format(l, k), key_r)
        bias_key = "encoder.transformer.{}.feed_forward.linear_{}.bias".format(l, k)
        B[bias_key] = mul_bias(A["encoder.transformer.{}.feed_forward.linear_{}_right_compactor.weight".format(l, k)],bias_key).squeeze(0).index_select(dim=0, index=indices_r.to(indices.device))
    for kk in ["gamma", "beta"]:
        key = "encoder.transformer.{}.layer_norm_2.{}".format(l, kk)
        B[key] = A[key].index_select(dim=0, index=indices_r.to(indices.device))




indices_r = mask_list["target.mlm_linear_1_right_compactor.weight"]
indices_l = mask_list["target.mlm_linear_1_left_compactor.weight"]
t_r = A["target.mlm_linear_1_right_compactor.weight"]
key_r = t_r.index_select(dim=0, index=indices_r.to(t_r.device))
t_l = A["target.mlm_linear_1_left_compactor.weight"]
key_l = t_l.index_select(dim=1, index=indices_l.to(t_l.device))
mul_l_r(key_l,"target.mlm_linear_1.weight", key_r)
bias_key = "target.mlm_linear_1.bias"
B["target.mlm_linear_1.bias"] = mul_bias(A["target.mlm_linear_1_right_compactor.weight"],bias_key).squeeze(0).index_select(dim=0, index=indices_r.to(A[bias_key].device))

bias_key = "target.layer_norm.gamma"
B[bias_key] = A[bias_key].index_select(dim=0, index=indices_r.to(A[bias_key].device))
bias_key = "target.layer_norm.beta"
B[bias_key] = A[bias_key].index_select(dim=0, index=indices_r.to(A[bias_key].device))

t_l = A["target.mlm_linear_2_left_compactor.weight"]
indices_l = mask_list["target.mlm_linear_2_left_compactor.weight"]
key_l = t_l.index_select(dim=1, index=indices_l.to(t_l.device))
B["target.mlm_linear_2.weight"] = A["target.mlm_linear_2.weight"].mm(key_l)
B["target.mlm_linear_2.bias"] = A["target.mlm_linear_2.bias"]

torch.save(B, f=args.output_model_path)
print('Done')