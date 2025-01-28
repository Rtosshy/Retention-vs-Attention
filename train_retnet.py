import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import load_dataset
import time
# import wandb
from preprocess import exec_preprocess
from retnet_mlm import RetNetMLM
from tqdm import tqdm  # tqdmをインポート
from icecream import ic

def train(epoch, model, optimizer, scheduler, train_dataloader, valid_dataloader):
    ic("Start training")
    step_ppls = []
    step_losses = []
    for i in range(epoch):
        model.train()
        print(f"Epoch {i}")
        start_time = time.time()
        total_train_loss = 0
        j = 0
        for batch in tqdm(train_dataloader, desc="Training", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # モデルの出力を取得
            logits = model(input_ids, attention_mask)

            # 損失を計算
            loss = model.compute_loss(logits, labels)
            step_losses.append(loss.item())
            total_train_loss += loss.item()
            step_ppl = torch.exp(torch.tensor(loss.item()))
            step_ppls.append(step_ppl)
            # print(f"Epoch{i} Step{j} PPL: {step_ppl}")

            # バックプロパゲーション
            loss.backward()
            optimizer.step()  # パラメータを更新
            optimizer.zero_grad()  # 勾配をゼロクリア
            scheduler.step()
            j += 1
        
        avg_epoch_loss = total_train_loss / len(train_dataloader)
        train_ppl = torch.exp(torch.tensor(avg_epoch_loss))
        print(f"Epoch {i} Train Loss: {avg_epoch_loss}")
        print(f"Epoch {i} Train PPL: {train_ppl}")
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch {i} Time: {epoch_time}")

        with open("epoch_times64ms5p5e_retnet.txt", "a") as f:
            print(epoch_time, file=f)

        # 検証フェーズ
        model.eval()  # モデルを評価モードに設定
        total_valid_loss = 0
        with torch.no_grad():  # 勾配計算を無効にする
            for batch in tqdm(valid_dataloader, desc="Validating", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids, attention_mask)
                loss = model.compute_loss(logits, labels)
                total_valid_loss += loss.item()

        avg_valid_loss = total_valid_loss / len(valid_dataloader)
        valid_ppl = torch.exp(torch.tensor(avg_valid_loss))
        print(f"Epoch {i} Validation Loss: {avg_valid_loss}")
        print(f"Epoch {i} Validation PPL: {valid_ppl}")

        torch.save(model.state_dict(), "model64ms5p5e_retnet.pth")
    
    with open("step_losses64ms5p5e_retnet.txt", "w") as f:
        for step_loss in step_losses:
            f.write(str(step_loss) + "\n")

    with open("step_ppls64ms5p5e_retnet.txt", "w") as f:
        for step_ppl in step_ppls:
            f.write(str(step_ppl.item()) + "\n")

def eval(model, test_dataloader):
    ic("Start testing")
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            loss = model.compute_loss(logits, labels)
            total_test_loss += loss.item()
    
    avg_test_loss = total_test_loss / len(test_dataloader)
    test_ppl = torch.exp(torch.tensor(avg_test_loss))
    print(f"Test PPL: {test_ppl}")

if __name__ == "__main__":
    # wandb.init(project="transformer-mlm")
    raw_datasets = load_dataset("bookcorpus", split="train[:5%]") # {text}
    checkpoint = "bert-base-uncased"
    # raw_datasets = raw_datasets.filter(lambda x: len(x["text"]) < 16)

    batch_size = 32
    max_seq_len = 64

    tokenizer, dataloaders = exec_preprocess(raw_datasets, checkpoint, max_seq_len, batch_size)

    vocab_size = tokenizer.vocab_size
    hidden_size = 512
    heads = 8
    layers = 6
    dropout = 0.1
    ic(max_seq_len)
    
    train_dataloader, valid_dataloader, test_dataloader = dataloaders

    model = RetNetMLM(vocab_size=vocab_size, hidden_size=hidden_size, heads=heads, layers=layers, max_seq_len=max_seq_len, dropout=dropout, double_v_dim=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_dataloader), eta_min=1e-6)

    train(5, model, optimizer, scheduler, train_dataloader, valid_dataloader)
    eval(model, test_dataloader)