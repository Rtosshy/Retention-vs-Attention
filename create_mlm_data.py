import torch

def create_mlm_data(input_ids, mask_token_id, vocab_size, mask_prob = 0.15):
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

    mask = (torch.rand(input_ids.shape) < mask_prob) & (input_ids != mask_token_id)

    masked_input_ids = input_ids.clone()
    masked_input_ids[mask] = mask_token_id

    mlm_labels = input_ids.clone()
    mlm_labels[~mask] = -100

    return masked_input_ids, mlm_labels


# テスト用のコード
if __name__ == "__main__":
    input_ids = torch.tensor([1, 2, 3, 4, 5])  # サンプルの入力ID
    mask_token_id = 0  # [MASK] トークンのID
    vocab_size = 6  # 語彙のサイズ
    masked_input_ids, mlm_labels = create_mlm_data(input_ids, mask_token_id, vocab_size)

    print("Masked Input IDs:", masked_input_ids)
    print("MLM Labels:", mlm_labels)