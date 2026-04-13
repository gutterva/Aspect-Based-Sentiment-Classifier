import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────
TEST_PATH             = "test.csv"
ASPECT_DETECTOR_DIR   = "aspect_detector"
SENTIMENT_CLASSIFIER_DIR = "sentiment_classifier"
MAX_LEN               = 128
BATCH_SIZE            = 32
RANDOM_SEED           = 42

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
ASPECT2IDX = {a: i for i, a in enumerate(ASPECTS)}

torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class AspectTestDataset(Dataset):
    def __init__(self, df, tokenizer):
        
        grouped      = df.groupby('text')['aspect'].apply(list).reset_index()
        self.texts   = grouped['text'].tolist()
        self.labels  = torch.tensor(
            [[1 if a in aspects else 0 for a in ASPECTS] for aspects in grouped['aspect']],
            dtype=torch.float
        )
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

class SentimentTestDataset(Dataset):
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


def evaluate_aspect_detector(test_df):
    print("\n" + "="*60)
    print("EVALUATING ASPECT DETECTOR")
    print("="*60)

    tokenizer  = RobertaTokenizer.from_pretrained(ASPECT_DETECTOR_DIR)
    model      = RobertaForSequenceClassification.from_pretrained(ASPECT_DETECTOR_DIR).to(device)
    model.eval()

    with open(f"{ASPECT_DETECTOR_DIR}/thresholds.json") as f:
        thresholds = json.load(f)

    test_ds     = AspectTestDataset(test_df, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.sigmoid(outputs.logits).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    
    tuned_preds = np.zeros_like(all_preds, dtype=int)
    for i, aspect in enumerate(ASPECTS):
        t = thresholds.get(aspect, 0.5)
        tuned_preds[:, i] = (all_preds[:, i] >= t).astype(int)

    print("\nPer-aspect classification report:")
    print(classification_report(all_labels, tuned_preds, target_names=ASPECTS, zero_division=0))
    print(f"Overall Macro F1  : {f1_score(all_labels, tuned_preds, average='macro',  zero_division=0):.4f}")
    print(f"Overall Micro F1  : {f1_score(all_labels, tuned_preds, average='micro',  zero_division=0):.4f}")
    print(f"Overall Sample F1 : {f1_score(all_labels, tuned_preds, average='samples',zero_division=0):.4f}")


def evaluate_sentiment_classifier(test_df):
    print("\n" + "="*60)
    print("EVALUATING SENTIMENT CLASSIFIER")
    print("="*60)

    tokenizer  = AutoTokenizer.from_pretrained(SENTIMENT_CLASSIFIER_DIR)
    model      = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_CLASSIFIER_DIR).to(device)
    model.eval()

    test_ds     = SentimentTestDataset(test_df, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
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

    print("\nOverall sentiment report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=['negative', 'positive'],
        zero_division=0
    ))

   
    print("\nPer-aspect sentiment F1:")
    test_df = test_df.copy()
    test_df['pred'] = all_preds
    for aspect in ASPECTS:
        sub = test_df[test_df['aspect'] == aspect]
        if len(sub) == 0:
            continue
        f1 = f1_score(sub['sentiment'].tolist(), sub['pred'].tolist(), average='macro', zero_division=0)
        print(f"  {aspect:<25} n={len(sub):>4}  macro F1: {f1:.4f}")

    
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['negative', 'positive'])
    disp.plot(cmap='Blues')
    plt.title("Sentiment Classifier — Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("\nConfusion matrix saved to confusion_matrix.png")


if __name__ == "__main__":
    test_df = pd.read_csv(TEST_PATH)
    print(f"Test set: {len(test_df)} rows")

    evaluate_aspect_detector(test_df)
    evaluate_sentiment_classifier(test_df)
