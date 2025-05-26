import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")


if __name__ == "__main__":
    text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)


    print(f"decoded text: {tokenizer.decode(integers)}")

    print(f"decoded text, but each integer is decoded separately:")
    for i in integers:
        print(tokenizer.decode([i]), end=" ")


# from what i understood, byte-pair encoding enables the LLM to be able to process any text, even if some words within the text did not appear in its training data. 
# it's a much more involved topic, this bpe. Let's check it out someday.