import torch

vocab_size = 6
output_dim = 3 

if __name__ == "__main__":
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(embedding_layer.weight) # prints the embedding 'layer'. it's a table of all possible embeddings of each token in the vocabulary. For token_id to embedding conversion, you just refer to embedding_layer.weight[token_id], giving you the tensor of embedding values for the token.


    print(embedding_layer(torch.tensor([3])))

    print(embedding_layer.weight[3]) # gives the same embedding, but mentions a different grad_fn, wonder why.



# the embedding layer is a matrix of size vocab_size x output_dim. Each token in vocab has a row for it, and output_dim number of dimensions to each token.

# hence, the embedding layer for token 3 is the 4th row of the embedding matrix (0-indexed)