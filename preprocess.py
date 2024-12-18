import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from tokenize_data import tokenize_data
from create_mlm_data import create_mlm_data
from icecream import ic


def exec_preprocess(raw_datasets, checkpoint, batch_size=8):
    tokenizer, tokenized_datasets = tokenize_data(raw_datasets, checkpoint)
    masked_input_ids, mlm_labels = create_mlm_data(tokenized_datasets['input_ids'], tokenizer.mask_token_id, tokenizer.pad_token_id)

    tokenized_dict = {
        'input_ids': masked_input_ids,
        'attention_mask': tokenized_datasets['attention_mask'],
        'labels': mlm_labels,
    }
    tokenized_datasets = Dataset.from_dict(tokenized_dict)
    tokenized_datasets.set_format("torch")

    train_testvalid = tokenized_datasets.train_test_split(test_size=0.2)
    valid_test = train_testvalid['test'].train_test_split(test_size=0.5)

    train_dataset = train_testvalid['train'] # {input_ids, attention_mask}
    valid_dataset = valid_test['train']
    test_dataset = valid_test['test']

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    dataloaders = [train_dataloader, valid_dataloader, test_dataloader]
    return tokenizer, dataloaders

# 例として、train_dataloaderからデータを取り出す
def check_dataloader(dataloader):
    for batch in dataloader:
        ic(batch['input_ids'])
        print(batch['labels'])
        break  # 最初のバッチだけを表示するためにbreakを使用

if __name__ == "__main__":
    raw_datasets = load_dataset("bookcorpus", split="train[:1%]") # {text}
    checkpoint = "bert-base-uncased"
    # raw_datasets = raw_datasets.select(range(0, 10))
    raw_datasets = raw_datasets.filter (lambda x: len(x["text"]) < 16)
    tokenizer, dataloaders = exec_preprocess(raw_datasets, checkpoint)
    train_dataloader, valid_dataloader, test_dataloader = dataloaders

    check_dataloader(train_dataloader)





    