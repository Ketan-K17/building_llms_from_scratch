# Now that we have created our the input and target dataset, we're almost ready to convert them into embeddings and feed it to an LLM to train on. 

# However, We cannot provide these datasets as just any Data structure such as lists of strings to the LLM, they have to be converted into Pytorch's Tensors.


import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt) #1
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk)) #2

    def __len__(self): #3
        return len(self.input_ids)

    def __getitem__(self, idx): #4
        return self.input_ids[idx], self.target_ids[idx]
    
#1 Tokenizes the entire text
#2 Uses a sliding window to chunk the book into overlapping sequences of max_length
#3 Returns the total number of rows in the dataset
#4 Returns a single row from the dataset

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2") #1
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) #2
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last, #3
        num_workers=num_workers #4
    )
    return dataloader

#1 Initializes the tokenizer
#2 Creates dataset
#3 drop_last=True drops the last batch if it is shorter than the specified batch_size to prevent loss spikes during training.
#4 The number of CPU processes to use for preprocessing


# there are max_length x tokens in a sequence
# batch_size x sequences in a batch
# stride is the number of tokens you shift between consecutive input sequences.