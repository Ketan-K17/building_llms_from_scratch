from creating_vocabulary import vocab

# we need to find a way of getting the token id when we pass the token as key. hence, we invert the vocabulary dictionary

# we put this in a nice tokenizer class.

import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab # our original vocabulary dict
        self.int_to_str = {i:s for s,i in vocab.items()} # inverted vocab dict

    # converts given text into a list of token ids
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    # converts a list of token ids into a text
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'\\])', r'\1', text)
        return text

# we'll use pre-made tokenizers soon, but know that most tokenizers have an encode() and decode() method.

if __name__ == "__main__":
    tokenizer = SimpleTokenizerV1(vocab)
    text = """"It's the last he painted, you know,"
    Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(f"token ids: {ids}")
    print(f"decoded text: {tokenizer.decode(ids)}")