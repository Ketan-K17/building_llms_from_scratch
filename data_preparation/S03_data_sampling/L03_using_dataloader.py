from L02_efficient_data_loader import create_dataloader_v1


with open("../the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

    
dataloader = create_dataloader_v1(raw_text, batch_size=1,max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader) #1
first_batch = next(data_iter)
print(first_batch)