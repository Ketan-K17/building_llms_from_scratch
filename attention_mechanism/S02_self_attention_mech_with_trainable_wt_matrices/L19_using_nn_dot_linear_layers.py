# We can improve the SelfAttention_v1 implementation further
# by utilizing PyTorch’s nn.Linear layers, which effectively
# perform matrix multiplication when the bias units are
# disabled. Additionally, a significant advantage of using
# nn.Linear instead of manually implementing
# nn.Parameter(torch.rand(...)) is that nn.Linear has an
# optimized weight initialization scheme, contributing to more
# stable and effective model training.

import torch
import torch.nn as nn

from attention_mechanism.S02_self_attention_mech_with_trainable_wt_matrices.L11_step0_calculating_3_vectors_for_all_tokens import inputs


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

if __name__ == "__main__":
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print(sa_v2(inputs))


# Note that SelfAttention_v1 and SelfAttention_v2 give different
# outputs because they use different initial weights for the
# weight matrices since nn.Linear uses a more sophisticated
# weight initialization scheme.