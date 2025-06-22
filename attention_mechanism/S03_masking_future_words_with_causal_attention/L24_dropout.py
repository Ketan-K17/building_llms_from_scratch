# This strategy is another minor tweak to the causal attention mechanism that is useful for reducing overfitting when training LLMs.


# Masking additional attention weights with
# dropout

# Dropout in deep learning is a technique where RANDOMLY
# selected hidden layer units are ignored during training,
# effectively “dropping” them out. This method helps prevent
# overfitting by ensuring that a model does not become overly
# reliant on any specific set of hidden layer units. It’s
# important to emphasize that dropout is only used during
# training and is disabled afterward.

# In the transformer architecture, including models like GPT,
# dropout in the attention mechanism is typically applied at
# two specific times: after calculating the attention weights or
# after applying the attention weights to the value vectors.
# Here we will apply the dropout mask after computing the
# attention weights, as illustrated in the figure below, 
# because it’s
# the more common variant in practice.

# NOTE: all in all, it's another mask where we just remove random weights from the tensor, just because.

import torch
from attention_mechanism.S03_masking_future_words_with_causal_attention.L23_softmax_for_renormalization import attn_weights

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5) # choosing a dropout rate of 50%
example = torch.ones(6, 6) # creating a tensor of ones


# applying the dropout mask to the attention weights
attn_weights_dropped = dropout(attn_weights)

if __name__ == "__main__":
    print(f"example: \n{example}")
    print(f"dropout(example): \n{dropout(example)}")

    # print(f"another dropout: \n{dropout(example)}")

    print(f"attn_weights_dropped: \n{attn_weights_dropped}")

    # NOTE: the dropout mask is random, so it will be different each time we run the code.


    # When applying dropout to an attention weight matrix with a
    # rate of 50%, half of the elements in the matrix are randomly
    # set to zero. To compensate for the reduction in active
    # elements, the values of the remaining elements in the
    # matrix are scaled up by a factor of 1/0.5 = 2. This scaling is
    # crucial to maintain the overall balance of the attention
    # weights, ensuring that the average influence of the attention
    # mechanism remains consistent during both the training and
    # inference phases.

