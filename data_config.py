import torch
from datasets import load_dataset

from preprocess import exec_preprocess
from icecream import ic

if __name__ == "__main__":
    raw_datasets = load_dataset("bookcorpus", split="train[:5%]")
    checkpoint = "bert-base-uncased"

    batch_size = 32
    max_seq_len = 128

    tokenizer, dataloaders = exec_preprocess(raw_datasets, checkpoint, max_seq_len, batch_size)

    num_steps = len(dataloaders[0]) / 32
    ic(num_steps)
    ic(max_seq_len)