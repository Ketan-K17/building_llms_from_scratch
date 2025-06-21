# this was an exercise to demonstrate that barring the initialization method of the weight matrices, the two classes are equivalent. the nn.linear method is better optimized for model training.

import torch.nn as nn
import torch

from attention_mechanism.S02_self_attention_mech_with_trainable_wt_matrices.L11_step0_calculating_3_vectors_for_all_tokens import inputs

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec
    
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
 
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
d_in = 3
d_out = 2
sa_v2 = SelfAttention_v2(d_in, d_out)
sa_v1 = SelfAttention_v1(d_in, d_out)

if __name__ == "__main__":

    print(sa_v2(inputs))

    sav2_w_query = sa_v2.W_query.weight.T
    sav2_w_key = sa_v2.W_key.weight.T
    sav2_w_value = sa_v2.W_value.weight.T

    sa_v1.W_query.data = sav2_w_query # sa_v1.W_query is a nn.Parameter object, which I understand is a regular tensor buth with a bunch of other metadata attached to it. sa_v1.W_query.data is the tensor part of the nn.Parameter object. hence when attributing another tensor to it, i have to write .data.
    sa_v1.W_key.data = sav2_w_key
    sa_v1.W_value.data = sav2_w_value

    print(sa_v1(inputs))
