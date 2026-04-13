import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

# ── Config ─────────────────────────────────────────────────────────────────
TRAIN_PATH      = r"C:\Users\your_path"
VAL_PATH        = r"C:\Users\your_path"
OUTPUT_DIR      = r"C:\Users\your_path"

MODEL_NAME  = "yangheng/deberta-v3-base-absa-v1.1"
MAX_LEN     = 128
BATCH_SIZE  = 16          
GRAD_ACCUM  = 4          
EPOCHS      = 10
LR          = 2e-5
WARMUP_RATIO= 0.1
EARLY_STOP  = 4
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts     = df['text'].tolist()
        self.aspects   = df['aspect'].tolist()
        self.labels    = torch.tensor(df['sentiment'].tolist(), dtype=torch.long)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        
        enc = self.tokenizer(
            self.texts[idx],
            self.aspects[idx],
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids'      : enc['input_ids'].squeeze(0),
            'attention_mask' : enc['attention_mask'].squeeze(0),
            'token_type_ids' : enc.get('token_type_ids', torch.zeros(MAX_LEN, dtype=torch.long)).squeeze(0),
            'labels'         : self.labels[idx]
        }


def train():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)

    print(f"  Train rows : {len(train_df)}")
    print(f"  Val   rows : {len(val_df)}")
    print(f"  Sentiment distribution (train):\n{train_df['sentiment'].value_counts()}")

    print(f"\nLoading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True
    ).to(device)

    train_ds = SentimentDataset(train_df, tokenizer)
    val_ds   = SentimentDataset(val_df,   tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    
    labels_array = train_df['sentiment'].values
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=labels_array
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"\nClass weights: neg={class_weights[0]:.4f}, pos={class_weights[1]:.4f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_loss  = float('inf')
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):
       
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            if step % 20 == 0:
                print(f"Epoch {epoch} | Step {step}/{len(train_loader)}")

            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            loss = criterion(outputs.logits, labels) / GRAD_ACCUM
            loss.backward()
            train_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()


        avg_train_loss = train_loss / len(train_loader)

        
        model.eval()
        val_loss   = 0.0
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i % 10 == 0:
                    print(f"Validation Step {i}/{len(val_loader)}")

                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels         = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )

                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        macro_f1     = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Macro F1: {macro_f1:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss  = avg_val_loss
            patience_count = 0
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"  ✓ Best model saved (val loss: {best_val_loss:.4f})")
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{EARLY_STOP})")
            if patience_count >= EARLY_STOP:
                print("Early stopping triggered.")
                break


    print("\nLoading best checkpoint for final val report...")
    model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR).to(device)
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels         = batch['labels']

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print(classification_report(
        all_labels, all_preds,
        target_names=['negative', 'positive'],
        zero_division=0
    ))

if __name__ == "__main__":
    train()
