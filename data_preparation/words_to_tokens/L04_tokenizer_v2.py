import re
from L03_adding_special_context_tokens import vocab

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    # converts given text into a list of token ids
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    # converts a list of token ids into a text
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'\\])', r'\1', text)
        return text

if __name__ == "__main__":
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(f"text to be tokenized:\n {text}")

    tokenizer = SimpleTokenizerV2(vocab)
    ids = tokenizer.encode(text)
    print(f"token ids: {ids}")
    print(f"decoded text: {tokenizer.decode(ids)}")


# notice how punctuation marks usually appear early in the vocab, hence, they have smaller token ids.

# A little bit more about tokenizers and GPT models: 

# The tokenizer used for GPT models only uses an <|endoftext|> token for
# simplicity. <|endoftext|> is analogous to the [EOS] token.
# <|endoftext|> is also used for padding. However, as we’ll
# explore in subsequent chapters, when training on batched
# inputs, we typically use a mask, meaning we don’t attend to
# padded tokens. Thus, the specific token chosen for padding
# becomes inconsequential.
# Moreover, the tokenizer used for GPT models also doesn’t
# use an <|unk|> token for out-of-vocabulary words. Instead,
# GPT models use a 'byte pair encoding' tokenizer, which breaks
# words down into subword units