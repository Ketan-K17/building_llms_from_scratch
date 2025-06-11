# STEP 2: We normalize the attention scores to get attention weights. Note that the sum of the weights is always 1.

import torch
from L05_step1_compute_attention_scores import attn_scores_2

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum() # normalizing attention scores to get attention weights.

if __name__ == "__main__":
    print("Attention weights:", attn_weights_2_tmp)
    print("Sum:", attn_weights_2_tmp.sum())