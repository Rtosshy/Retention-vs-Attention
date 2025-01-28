import torch
import time
import psutil
from transformers import BertTokenizer
from icecream import ic
from datasets import Dataset
from retnet_mlm import RetNetMLM
from create_mlm_data import create_mlm_data
from tokenize_data import tokenize_data

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(checkpoint)

    vocab_size = tokenizer.vocab_size
    max_seq_len = 64
    hidden_size = 512
    heads = 8
    layers = 6
    dropout = 0.1
    batch_size = 32
    model = RetNetMLM(vocab_size=vocab_size, hidden_size=hidden_size, heads=heads, layers=layers, max_seq_len=max_seq_len, dropout=dropout, double_v_dim=False)
    model.to(device)
    model.load_state_dict(torch.load("retnet_models/model64ms5p5e_retnet.pth"))

    raw_strings = [
        'usually , he would be tearing around the living room , playing with his toys .',
        'the only thing that kept him from doing that was the fact that he was sick .',
    ]

    raw_datasets = Dataset.from_dict(
        {
            "text": raw_strings
        }
    )

    _, tokenized_datasets = tokenize_data(raw_datasets, checkpoint)
    masked_input_ids, mlm_labels = create_mlm_data(tokenized_datasets['input_ids'], tokenizer.mask_token_id, tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id)

    masked_positions = (mlm_labels != -100)

    # 並列形態
    with torch.no_grad():
        
        masked_input_ids = masked_input_ids.to(device)
        attention_mask = tokenized_datasets['attention_mask'].to(device)

        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 ** 2

        start_time = time.time()

        logits = model(masked_input_ids, attention_mask)

        end_time = time.time()

        memory_after = process.memory_info().rss / 1024 ** 2
        inference_time = end_time - start_time

    print(f"推論時間: {inference_time:.4f}秒")
    print(f"メモリ使用量: {memory_after - memory_before:.2f} MB")

    masked_logits = logits[masked_positions]
    correct_ids = mlm_labels[masked_positions]

    _, topk_ids = torch.topk(masked_logits, k=10, dim=-1)
    ic(torch.sort(torch.softmax(masked_logits, dim=-1)[0], descending=True)[:10])
    ic(topk_ids.shape[0])
    for masked_input_id in masked_input_ids:
        ic(tokenizer.decode(masked_input_id))
    
    output_info = []

    for i in range(len(topk_ids)):
        output_info.append(
            {
                "topk_ids": tokenizer.decode(topk_ids[i]),
                "correct_ids": tokenizer.decode(correct_ids[i])
            }
        )
    ic(output_info)

    # 再帰形態
    # with torch.no_grad():
    #     masked_input_ids = masked_input_ids.to(device)
    #     process = psutil.Process()

    #     s_n_1s = torch.zeros((batch_size, hidden_size//heads, hidden_size//heads)).to(device)

    #     for n in range(max_seq_len):
    #         memory_before = process.memory_info().rss
    #         start_time = time.time()
    #         logits, s_ns = model.forward_recurrent(input_ids=masked_input_ids[:, n:n+1], s_n_1s=s_n_1s, n=n)
    #         end_time = time.time()
    #         s_n_1s = s_ns
    #         memory_after = process.memory_info().rss
    #         print(f"メモリ使用量: {memory_after - memory_before:.2f} B")
    #         inference_time = end_time - start_time
    #         print(f"推論時間: {inference_time:.4f}秒")
        
        

        
        
    
    
    
        





    