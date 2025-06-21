# Step 3: computing the context vector using the weighted sum of the values with attention weights.



import torch
from attention_mechanism.S02_self_attention_mech_with_trainable_wt_matrices.L11_step0_calculating_3_vectors_for_all_tokens import keys, values
from attention_mechanism.S02_self_attention_mech_with_trainable_wt_matrices.L14_step2_normalization_to_attn_wts import attn_weights_2


context_vec_2 = attn_weights_2 @ values


if __name__ == "__main__":
    print(context_vec_2)
    print(context_vec_2.shape)

