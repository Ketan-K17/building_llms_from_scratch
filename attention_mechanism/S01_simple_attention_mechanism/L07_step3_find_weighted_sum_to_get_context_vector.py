# STEP 3: We now compute the context vector for the query token, by taking a weighted sum of the input tokens against the attention weights.

import torch
from L06_step2_normalize_with_softmax import attn_weights_2
from L05_step1_compute_attention_scores import inputs

query = inputs[1] # our query token, that we'll compute the context vector for.

context_vec_2 = torch.zeros(query.shape) # initialize the context vector to 0.
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i # weighted sum of the input tokens, to get the context vector.

if __name__ == "__main__":
    print(context_vec_2)


# note: the next thing to do is figure out an implementation for doing this for all input tokens, simulateneously.


