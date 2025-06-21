# Step 1: computing attention scores 
# -> to compute attention scores, take dot product of query vector of the query token with all tokens' key vectors.



# we'll find attention score w22. (q2 against k2)
import torch
from attention_mechanism.S02_self_attention_mech_with_trainable_wt_matrices.L11_step0_calculating_3_vectors_for_all_tokens import keys, values, query_2


keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)


# computing all of the attention scores for a given query vector all at once.
attn_scores_2 = query_2 @ keys.T #1

if __name__ == "__main__":
    print(attn_score_22)

    print("attention scores for token 2 against all tokens")
    print(attn_scores_2)