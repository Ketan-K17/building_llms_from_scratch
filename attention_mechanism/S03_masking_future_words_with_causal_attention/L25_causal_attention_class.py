# "let’s ensure that the code can handle
# batches consisting of more than one input so that the
# CausalAttention class supports the batch outputs produced by
# the data loader we implemented in chapter 2."

# He said this before moving onto creating the CausalAttention class (page 155)... i'm not entirely sure what he meant by that. As in... I've probably forgotten what he taught in chapter 2.

import torch
import torch.nn as nn
from attention_mechanism.S02_self_attention_mech_with_trainable_wt_matrices.L11_step0_calculating_3_vectors_for_all_tokens import inputs, d_in, d_out


# "For simplicity, to simulate such batch inputs, we duplicate
# the input text example:"

batch = torch.stack((inputs, inputs), dim=0) # this creates a batch of 2 inputs.

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) #1
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        ) #2

    def forward(self, x):
        b, num_tokens, d_in = x.shape #3
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2) # note, always use .transpose over .T, since .T works properly only for 1D or 2D tensors, and .transpose works for any number of dimensions. 
        attn_scores.masked_fill_( #4
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) 
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec
    
#1 Compared to the previous SelfAttention_v1 class, we added a dropout layer.
#2 The register_buffer call is also a new addition (more information is provided in the following text).
#3 We transpose dimensions 1 and 2, keeping the batch dimension at the first position (0).
#4 In PyTorch, operations with a trailing underscore are performed inplace, avoiding unnecessary memory copies

if __name__ == "__main__":
    print(f"batch: \n{batch}")
    print(f"batch.shape: \n{batch.shape}")
    # in batch.shape, first num = no. of individual tensors, 2nd num = no. of tokens (i.e. rows in tensor), 3rd num = no. of dimensions (i.e. columns in tensor).


    torch.manual_seed(123)
    context_length = batch.shape[1]
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vecs = ca(batch)
    print("context_vecs.shape:", context_vecs.shape)




