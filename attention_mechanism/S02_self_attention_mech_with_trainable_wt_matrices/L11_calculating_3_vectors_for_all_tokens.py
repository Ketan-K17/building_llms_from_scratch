# when working with self-attention using trainable weight matrices, there's a step 0, where you compute the 3 vectors for all tokens. It involves 3 weight matrices, and the input tokens.

import torch
from attention_mechanism.S01_simple_attention_mechanism.L05_step1_compute_attention_scores import inputs

x_2 = inputs[1] # The second input element
d_in = inputs.shape[1] # The input embedding size, d=3
d_out = 2 # The output embedding size, d_out=2

# NOTE that in GPT-like models, the input and output dimensions are usually the same, but to better follow the computation, we’ll use different input (d_in=3) and output (d_out=2) dimensions here.


# initializing the weight matrices
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# We set requires_grad=False to reduce clutter in the outputs,
# but if we were to use the weight matrices for model training,
# we would set requires_grad=True to update these matrices
# during model training.

# compute the 3 vectors for the second input element
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

# compute the key and value vectors for all input elements (you need all of them to find the context vector for even one input token)
keys = inputs @ W_key
values = inputs @ W_value


if __name__ == "__main__":
    print(f"query_2: {query_2}")
    print(f"key_2: {key_2}")
    print(f"value_2: {value_2}")
    print("keys.shape:", keys.shape)
    print("values.shape:", values.shape)