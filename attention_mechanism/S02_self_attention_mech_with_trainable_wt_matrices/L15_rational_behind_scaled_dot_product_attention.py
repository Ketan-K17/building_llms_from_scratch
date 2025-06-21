# THE RATIONALE BEHIND SCALED-DOT PRODUCT ATTENTION

# The reason for the normalization by the embedding
# dimension size is to improve the training performance by
# avoiding small gradients. For instance, when scaling up the
# embedding dimension, which is typically greater than
# 1,000 for GPT-like LLMs, large dot products can result in
# very small gradients during backpropagation due to the
# softmax function applied to them. As dot products
# increase, the softmax function behaves more like a step
# function, resulting in gradients nearing zero. These small
# gradients can drastically slow down learning or cause
# training to stagnate.
# The scaling by the square root of the embedding
# dimension is the reason why this self-attention mechanism
# is also called scaled-dot product attention.