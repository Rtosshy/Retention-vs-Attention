import torch
from datasets import load_dataset
from preprocess import exec_preprocess
from icecream import ic

def create_mlm_data(input_ids, mask_token_id, pad_token_id, mask_prob = 0.15):
    """
    Create masked input and labels for MLM training

    Args:
        input_ids: Original input ids
        mask_token_id: ID of [MASK] token
        vocab_size: Size of vocabulary
        mask_prob: Probability of masking token
    
    Returns:
        masked_input_ids: Input with masks
        mlm_labels: Labels for masked tokens (-100 for unmasked positions)
    """

    mask = (torch.rand(input_ids.shape) < mask_prob) & (input_ids != mask_token_id) & (input_ids != pad_token_id)

    masked_input_ids = input_ids.clone()
    masked_input_ids[mask] = mask_token_id

    mlm_labels = input_ids.clone().float()
    mlm_labels[~mask] = -100

    return masked_input_ids, mlm_labels


# テスト用のコード
if __name__ == "__main__":
    raw_datasets = load_dataset("bookcorpus", split="train[:1%]") # {text}
    checkpoint = "bert-base-uncased"
    tokenizer, dataloaders = exec_preprocess(raw_datasets, checkpoint)
    train_dataloader, _, _ = dataloaders

    batch = next(iter(train_dataloader))
    input_ids = batch['input_ids'][0]
    ic(tokenizer.decode(input_ids))
    ic(input_ids)
    mask_token_id = tokenizer.mask_token_id  # [MASK] トークンのID
    ic(mask_token_id)
    pad_token_id = tokenizer.pad_token_id  # [PAD] トークンのID
    ic(pad_token_id)
    masked_input_ids, mlm_labels = create_mlm_data(input_ids, mask_token_id, pad_token_id)

    ic(masked_input_ids)
    ic(mlm_labels)