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