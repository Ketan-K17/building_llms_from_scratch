from L01_tokenizer_v1 import SimpleTokenizerV1
from creating_vocabulary import vocab

if __name__ == "__main__":
    tokenizer = SimpleTokenizerV1(vocab)
    text = "Hello, do you like tea?"
    print(tokenizer.encode(text))


# this gives off a key error on the word "hello" because it's not in the vocabulary, i.e. the word hello was not used in the verdict.txt file.
# this highlights the importance of using a large corpus of text to create a vocabulary.