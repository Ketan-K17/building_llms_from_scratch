import re
with open("../the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

vocab = {token:integer for integer,token in enumerate(all_words)}

if __name__ == "__main__":
    print(f"vocab size: {vocab_size}")
    for i, item in enumerate(vocab.items()):
        print(item)
        if i >= 50:
            break
