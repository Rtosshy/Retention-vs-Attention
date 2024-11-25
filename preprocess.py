from datasets import load_dataset
from transformers import BertTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("bookcorpus", split="train[:10%]")
checkpoint = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_testvalid = tokenized_datasets.train_test_split(test_size=0.2)
valid_test = train_testvalid['test'].train_test_split(test_size=0.5)

train_dataset = train_testvalid['train'] # {text, inpt_ids, token_type_ids, attention_mask}
valid_dataset = valid_test['train']
test_dataset = valid_test['test']