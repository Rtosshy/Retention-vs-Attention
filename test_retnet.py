import torch
from datasets import load_dataset

from preprocess import exec_preprocess
from retnet_mlm import RetNetMLM
from tqdm import tqdm
from icecream import ic

def eval(model, test_dataloader):
    ic("Start testing")
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            loss = model.compute_loss(logits, labels)
            total_test_loss += loss.item()
    
    avg_test_loss = total_test_loss / len(test_dataloader)
    test_ppl = torch.exp(torch.tensor(avg_test_loss))
    print(f"Test PPL: {test_ppl}")

def eval_reccurent(model, test_dataloader):
    ic("Start testing")
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing", leave=False):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            s_n_1s = torch.zeros((batch_size, hidden_size//heads, hidden_size//heads)).to(device)

            for n in range(max_seq_len):
                logits, s_ns = model.forward_recurrent(input_ids=input_ids[:, n:n+1], s_n_1s=s_n_1s, n=n)
                s_n_1s = s_ns
            # loss = model.compute_loss(logits, labels)
            # total_test_loss += loss.item()

if __name__ == "__main__":
    raw_datasets = load_dataset("bookcorpus", split="train[:5%]") # {text}
    checkpoint = "bert-base-uncased"
    # raw_datasets = raw_datasets.filter(lambda x: len(x["text"]) < 16)

    batch_size = 32
    max_seq_len = 64

    tokenizer, dataloaders = exec_preprocess(raw_datasets, checkpoint, max_seq_len, batch_size)

    vocab_size = tokenizer.vocab_size
    hidden_size = 512
    heads = 8
    layers = 6
    dropout = 0.1
    ic(max_seq_len)
    
    _, _, test_dataloader = dataloaders

    model = RetNetMLM(vocab_size=vocab_size, hidden_size=hidden_size, heads=heads, layers=layers, max_seq_len=max_seq_len, dropout=dropout, double_v_dim=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load("retnet_models/model64ms5p5e_retnet.pth"))
    eval_reccurent(model, test_dataloader)
