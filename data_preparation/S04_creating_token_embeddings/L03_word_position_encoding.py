# page 99

# the weight matrix of the embedding layer we created does not care about where the token appears in a sequence. The same token appearing in different positions in a sequence will have the same embedding.

# Apparently, the self-attention mechanism of the LLM too, does not care about token position. turns out that... What we have rn, the embedding weight matrix i mean.. is enough, technically, but it's better to make it aware of token position too. 
# there's no concrete reason attributed to this pun. 


# there's 2 ways to make the embedding layer aware of token position.
# 1. Absolute positional embedding
# 2. Relative positional embedding


# absolute mein a unique embedding dimension value is added to each dimension of the token embedding (shown in image)

# relative apparently has been called better sometimes because it's more general across different sequence lengths. So even if you get sequence lengths that are unheard of, it can perform decently well. 

# fun fact: GPT models using absolute positional encoding apparently, only thing is they don't remain static, they get optimized during pretraining.