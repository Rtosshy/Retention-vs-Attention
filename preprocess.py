from datasets import load_dataset
from transformers import BertTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from icecream import ic

def exec_preprocess(raw_datasets, checkpoint):

    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True) # {text, input_ids, token_type_ids, attention_mask}
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    tokenized_datasets = tokenized_datasets.remove_columns(["text", "token_type_ids"])
    tokenized_datasets.set_format("torch")

    train_testvalid = tokenized_datasets.train_test_split(test_size=0.2)
    valid_test = train_testvalid['test'].train_test_split(test_size=0.5)

    train_dataset = train_testvalid['train'] # {input_ids, attention_mask}
    valid_dataset = valid_test['train']
    test_dataset = valid_test['test']

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_dataset, batch_size=8, collate_fn=data_collator)
    test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=data_collator)

    dataloaders = [train_dataloader, valid_dataloader, test_dataloader]
    return tokenizer, dataloaders

if __name__ == "__main__":
    raw_datasets = load_dataset("bookcorpus", split="train[:1%]") # {text}
    checkpoint = "bert-base-uncased"
    tokenizer, dataloaders = exec_preprocess(raw_datasets, checkpoint)
    train_dataloader, _, _ = dataloaders

    batch = next(iter(train_dataloader))
    ic(batch.keys())
    ic('input_ids')
    ic(batch['input_ids'][0])
    ic('attention_mask')
    ic(batch['attention_mask'][0])