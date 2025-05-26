# There are special context tokens that need to be added to our tokenizer too to consider some special cases - 

# <|unk|> - unknown token, for a token within a text that is not recognized by the tokenizer.
# <|endoftext|> - helps during pretraining, to differenttate between unrelated texts that have been concatenanted.

from creating_vocabulary import all_words

all_words.extend(["<|unk|>", "<|endoftext|>"])
vocab = {token:integer for integer,token in enumerate(all_words)}

if __name__ == "__main__":
    print(f"length of extended vocab with special tokens: {len(vocab)}")
    print("last 5 items in the extended vocab:")
    for i, item in enumerate(list(vocab.items())[-5:]):
        print(item)