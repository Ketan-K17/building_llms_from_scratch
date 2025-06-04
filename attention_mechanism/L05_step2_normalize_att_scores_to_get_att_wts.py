import torch
from L04_step1_compute_attention_scores import attn_scores_2

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum() # normalizing attention scores to get attention weights.

if __name__ == "__main__":
    print("Attention weights:", attn_weights_2_tmp)
    print("Sum:", attn_weights_2_tmp.sum())