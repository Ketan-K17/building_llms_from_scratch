# Now, we want to go from the attention scores to the
# attention weights. We compute
# the attention weights by scaling the attention scores and
# using the softmax function. However, now we scale the
# attention scores by dividing them by the square root of the
# embedding dimension of the keys (taking the square root is
# mathematically the same as exponentiating by 0.5):

import torch
from attention_mechanism.S02_self_attention_mech_with_trainable_wt_matrices.L13_step1_computing_attention_scores import attn_scores_2

from attention_mechanism.S02_self_attention_mech_with_trainable_wt_matrices.L11_step0_calculating_3_vectors_for_all_tokens import keys

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)


if __name__ == "__main__":
    print(attn_weights_2)
    print(attn_weights_2.shape)