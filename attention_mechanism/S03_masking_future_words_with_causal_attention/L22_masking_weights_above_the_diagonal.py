# from what i understand, given the regular attention scores table, you do the following to get the causal attention weights table:

# 1. normalize the current attention scores table to get the attention weights table
# 2. set the weights above the diagonal to 0
# 3. renormalize the attention weights table to get the causal attention weights table. However, for renormalization, you cannot be using softmax just like that, since even the masked weights would still contribute e^0 = 1 to denominator of the softmax function.


import torch
from attention_mechanism.S02_self_attention_mech_with_trainable_wt_matrices.L11_step0_calculating_3_vectors_for_all_tokens import inputs
from attention_mechanism.S02_self_attention_mech_with_trainable_wt_matrices.L20_comparing_both_classes import sa_v2, sa_v1


queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T

# STEP 1.
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) # this is the regular attention weights tensor. it is a tensor of shape (context_length, context_length)

context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length)) # torch.tril (i.e. torch.TRIangle_Lower) creates a lower triangular matrix of ones with the specified shape. This is going to be our mask.

if __name__ == "__main__":
    print(f"original attn_weights: \n{attn_weights}")
    print(f"context_length: {context_length}")
    print(f"mask_simple: \n{mask_simple}")

    # STEP 2.
    masked_simple = attn_weights*mask_simple # this is the unnormalized causal attention weights tensor. it is a tensor of shape (context_length, context_length). You get it by multiplying the original attention weights tensor with the mask.
    print(f"masked_simple after multiplying with attention weights: \n{masked_simple}")

    # STEP 3.
    row_sums = masked_simple.sum(dim=-1, keepdim=True) # tensor of shape (context_length, 1), with each entry being sum of the row.
    masked_simple_norm = masked_simple / row_sums # this is the normalized causal attention weights tensor. it is a tensor of shape (context_length, context_length).
    print(f"masked_simple_norm: \n{masked_simple_norm}")

# NOTE: you'd use the torch.triu() function to get the upper triangular matrix of ones with the specified shape.



