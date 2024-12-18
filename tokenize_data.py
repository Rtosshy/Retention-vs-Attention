from datasets import load_dataset
from transformers import BertTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from icecream import ic

def tokenize_data(raw_datasets, checkpoint):

    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding='max_length', max_length=16)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True) # {text, input_ids, token_type_ids, attention_mask}
    
    # バッチごとにpaddingを行う場合
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_datasets = tokenized_datasets.remove_columns(["text", "token_type_ids"])
    tokenized_datasets.set_format("torch")


    return tokenizer, tokenized_datasets

if __name__ == "__main__":
    raw_datasets = load_dataset("bookcorpus", split="train[:1%]") # {text}
    checkpoint = "bert-base-uncased"
    # raw_datasets = raw_datasets.select(range(0, 10))
    raw_datasets = raw_datasets.filter (lambda x: len(x["text"]) < 16)
    tokenizer, tokenized_datasets = tokenize_data(raw_datasets, checkpoint)