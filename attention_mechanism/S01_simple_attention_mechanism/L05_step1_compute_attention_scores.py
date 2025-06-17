# We will now implement a simple version of the self-attention mechanism, one that does not use 'trainable weights'.
# Technically, the goal of the self-attention mechanism is to compute a context vector for a given query token, by attending to all the input tokens. This is done for every input token.

# STEP 1: Given the embeddings of each token in the input sequence, we choose one token (query token) and compute its attention scores against all other tokens in the input sequence (including itself). 
# Attention scores are computed by taking the dot product between the query token and each of the input tokens.

import torch

# List of input token embeddings already handed to us.
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x1)
    [0.55, 0.87, 0.66], # journey (x2)
    [0.57, 0.85, 0.64], # starts (x3)
    [0.22, 0.58, 0.33], # with (x4)
    [0.77, 0.25, 0.10], # one (x5)
    [0.05, 0.80, 0.55]] # step (x6)
)

query = inputs[1] # (journey) our query token, that we'll compute the context vector for.
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)


if __name__ == "__main__":
    # attention scores for the query token, with respect to all the input tokens.
    print(attn_scores_2) 


# note: the dot product between the query token and input token is a measure of how relevant the input token is to the query token. Higher the w score, more relevant the input token is to the query token.