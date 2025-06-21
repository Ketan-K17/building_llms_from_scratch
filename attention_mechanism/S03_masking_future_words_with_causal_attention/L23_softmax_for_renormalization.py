# We discussed in the previous file why we cannot use softmax just like that for renormalization. We need to make a tiny modification to our mask if we want to use softmax for renormalization.


# In step 2, instead of masking the upper triangular attn_score values with 0's, we mask them with -inf. since e^(-inf) = 0, this will make the softmax function ignore the masked values.


import torch
from attention_mechanism.S03_masking_future_words_with_causal_attention.L22_masking_weights_above_the_diagonal import attn_scores, context_length, keys

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)


attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1) # this is the renormalized causal attention weights tensor of shape (context_length, context_length).

if __name__ == "__main__":
    print(f"upper triangular mask: \n{mask}")
    print(f"masked attn_scores: \n{masked}")
    print(f"renormalized causal attention weights: \n{attn_weights}")