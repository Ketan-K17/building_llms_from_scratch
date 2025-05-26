import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

with open("../the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    

if __name__ == "__main__":
    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))

    # got rid of the first 50 tokens to make a more interesting-looking input-output pair (otherwise it would have been all punctuation marks since they appear at the beginning of the vocab as you now know.)
    enc_sample = enc_text[50:]


    context_size = 4 # context size determines how many tokens are used to predict the next token.
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size+1]
    print(f"x: {x}")
    print(f"y:      {y}\n")

    # prints all input tokens ----> the target token.
    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(context, "---->", desired)


    # the same represented in terms of words.
    print("\n")
    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))