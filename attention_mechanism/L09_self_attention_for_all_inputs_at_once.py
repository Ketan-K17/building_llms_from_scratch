# Calculating context vectors for all input tokens at once.

import torch
from L05_step1_compute_attention_scores import inputs

# STEP 1.
attn_scores = torch.empty(6, 6)
# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attn_scores[i, j] = torch.dot(x_i, x_j)

attn_scores = inputs @ inputs.T # matrix multiplication of the input tokens with their transposed version.

# STEP 2.
attn_weights = torch.softmax(attn_scores, dim=-1)

# STEP 3.
context_vecs = attn_weights @ inputs


if __name__ == "__main__":
    print("attention scores: \n", attn_scores)
    print("attention weights: \n", attn_weights)

    print("sum of attention weights: ", attn_weights.sum(dim=-1))

    print("context vectors: \n", context_vecs)



# NOTE: while the for loop code and matrix multiplication code give the same result, the matrix multiplication code is faster and just better to use.


# NOTE(2): Notice the tensor, you'll find that it's symmetric. This is because as it stands now, attention score between x_i and x_j (x_i as query token) is the same as the attention score between x_j and x_i (x_j as query token).
# This is going to change, when we add trainable weights to the self-attention mechanism.


# NOTE(3): dim = -1 means function operates along the last dimension of the tensor (in this case it's along the columns).