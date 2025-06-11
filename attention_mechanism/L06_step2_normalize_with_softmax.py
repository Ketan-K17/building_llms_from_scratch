import torch
from L05_step1_compute_attention_scores import attn_scores_2

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

# attn_weights_2_naive = softmax_naive(attn_scores_2)
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
if __name__ == "__main__":
    print("Attention weights:", attn_weights_2)
    print("Sum:", attn_weights_2.sum())


# we must prioritise using softmax for normalization. Offers more favourable gradient properties it seems.