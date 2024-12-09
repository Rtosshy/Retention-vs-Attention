from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

print(raw_datasets["train"][0])


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print(tokenized_datasets["train"][0])

tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
print(tokenized_datasets["train"][0])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
print(tokenized_datasets["train"][0])
tokenized_datasets.set_format("torch")
print(tokenized_datasets["train"].column_names)

print(tokenized_datasets["train"][0])

from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

for batch in train_dataloader:
    break

print({k: v.shape for k, v in batch.items()})