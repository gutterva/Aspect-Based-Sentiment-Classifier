import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report
import json


TRAIN_PATH      = r"C:\Users\athar\Desktop\ABSA_NLP\train.csv"
VAL_PATH        = r"C:\Users\athar\Desktop\ABSA_NLP\val.csv"
OUTPUT_DIR      = r"C:\Users\athar\Desktop\ABSA_NLP\aspect_detector"

MODEL_NAME      = "roberta-base"
MAX_LEN         = 128
BATCH_SIZE      = 32
GRAD_ACCUM      = 2     
EPOCHS          = 10
LR              = 2e-5
WARMUP_RATIO    = 0.1
EARLY_STOP      = 4       
RANDOM_SEED     = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


ASPECTS = [
    "app performance",
    "brand satisfaction",
    "ease of use",
    "support attitude",
    "pricing value",
    "delivery speed",
    "food quality",
    "competitor comparison",
    "pricing discounts",
    "account access",
]
NUM_ASPECTS = len(ASPECTS)
ASPECT2IDX  = {a: i for i, a in enumerate(ASPECTS)}

def build_multilabel_df(path):
    """
    Input  : exploded CSV (one row per sentence-aspect pair)
    Output : one row per unique sentence with a binary vector of aspects
    """
    df = pd.read_csv(path)
   
    grouped = df.groupby('text')['aspect'].apply(list).reset_index()
    grouped.columns = ['text', 'aspects']

  
    def to_vector(aspect_list):
        vec = [0] * NUM_ASPECTS
        for a in aspect_list:
            if a in ASPECT2IDX:
                vec[ASPECT2IDX[a]] = 1
        return vec

    grouped['labels'] = grouped['aspects'].apply(to_vector)
    return grouped


class AspectDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts     = df['text'].tolist()
        self.labels    = torch.tensor(df['labels'].tolist(), dtype=torch.float)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids'      : enc['input_ids'].squeeze(0),
            'attention_mask' : enc['attention_mask'].squeeze(0),
            'labels'         : self.labels[idx]
        }


def compute_pos_weights(df):
    """
    For each aspect, pos_weight = (# negative samples) / (# positive samples)
    This handles per-aspect class imbalance inside BCEWithLogitsLoss.
    """
    label_matrix = np.array(df['labels'].tolist())
    pos          = label_matrix.sum(axis=0)
    neg          = len(label_matrix) - pos
    
    pos_weight   = neg / np.clip(pos, 1, None)
    return torch.tensor(pos_weight, dtype=torch.float).to(device)


def train():
    print("Building datasets...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    train_df  = build_multilabel_df(TRAIN_PATH)
    val_df    = build_multilabel_df(VAL_PATH)

    print(f"  Train sentences : {len(train_df)}")
    print(f"  Val   sentences : {len(val_df)}")
    print(f"  Aspects         : {ASPECTS}")

    train_ds = AspectDataset(train_df, tokenizer)
    val_ds   = AspectDataset(val_df,   tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_ASPECTS,
        problem_type="multi_label_classification"
    ).to(device)

    pos_weights = compute_pos_weights(train_df)
    criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps= int(total_steps * WARMUP_RATIO)
    scheduler   = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_loss   = float('inf')
    patience_count  = 0

    for epoch in range(1, EPOCHS + 1):
  
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = criterion(outputs.logits, labels) / GRAD_ACCUM
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
            for batch in val_loader:
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels         = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss    = criterion(outputs.logits, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs.logits).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        all_preds    = np.vstack(all_preds)
        all_labels   = np.vstack(all_labels)

   
        preds_binary = (all_preds >= 0.5).astype(int)
        macro_f1     = f1_score(all_labels, preds_binary, average='macro', zero_division=0)

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Macro F1 (0.5): {macro_f1:.4f}")

      
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

    
    print("\nTuning per-aspect thresholds on validation set...")
    model = RobertaForSequenceClassification.from_pretrained(OUTPUT_DIR).to(device)
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.sigmoid(outputs.logits).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

 
    thresholds = {}
    for i, aspect in enumerate(ASPECTS):
        best_t  = 0.5
        best_f1 = 0.0
        for t in np.arange(0.1, 0.91, 0.05):
            preds = (all_preds[:, i] >= t).astype(int)
            f1    = f1_score(all_labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t  = round(float(t), 2)
        thresholds[aspect] = best_t
        print(f"  {aspect:<25} → threshold: {best_t:.2f}  (val F1: {best_f1:.4f})")

    with open(os.path.join(OUTPUT_DIR, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nThresholds saved to {OUTPUT_DIR}/thresholds.json")

    print("\nFinal val report with tuned thresholds:")
    tuned_preds = np.zeros_like(all_preds, dtype=int)
    for i, aspect in enumerate(ASPECTS):
        tuned_preds[:, i] = (all_preds[:, i] >= thresholds[aspect]).astype(int)

    print(classification_report(
        all_labels, tuned_preds,
        target_names=ASPECTS, zero_division=0
    ))

if __name__ == "__main__":
    train()
