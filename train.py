import torch
from torch.optim import AdamW
from datasets import load_dataset
from preprocess import exec_preprocess
from transformer_mlm import TransformerMLM

raw_datasets = load_dataset("bookcorpus", split="train[:1%]") # {text}
checkpoint = "bert-base-uncased"
raw_datasets = raw_datasets.filter(lambda x: len(x["text"]) < 16)

tokenizer, dataloaders = exec_preprocess(raw_datasets, checkpoint)

train_dataloader, valid_dataloader, test_dataloader = dataloaders

vocab_size = tokenizer.vocab_size
d_model = 512
n_heads = 8
n_layers = 6
max_seq_len = 16 # tokenizer.model_max_length
batch_size = 8

model = TransformerMLM(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers, max_seq_len=max_seq_len)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

def train(epoch, model, optimizer, train_dataloader, valid_dataloader):
    model.train()
    for i in range(epoch):
        print(f"Epoch {i}")
        j = 0
        for batch in train_dataloader:
            print(f"Batch {j}")
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)

            loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch {i} Batch {j} Loss: {loss.item()}")
            j += 1

        print(f"Epoch {i} Loss: {loss.item()}")
    model.eval()


train(1, model, optimizer, train_dataloader, valid_dataloader)