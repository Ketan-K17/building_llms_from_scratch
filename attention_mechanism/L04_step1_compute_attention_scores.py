import torch

# List of input token embeddings already handed to us.
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66], # journey (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]] # step (x^6)
)


query = inputs[1] # our query token, that we'll compute the context vector for.
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)


if __name__ == "__main__":
    # attention scores for the query token, with respect to all the input tokens.
    print(attn_scores_2) 


# note: the dot product between the query token and input token is a measure of how relevant the input token is to the query token. Higher the w score, more relevant the input token is to the query token.