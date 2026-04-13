import json
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification


ASPECT_DETECTOR_DIR      = r"C:\Users\your_path"
SENTIMENT_CLASSIFIER_DIR = r"C:\Users\your_path"
MAX_LEN                  = 128

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

SENTIMENT_MAP = {0: "negative", 1: "positive"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Loading aspect detector...")
aspect_tokenizer = RobertaTokenizer.from_pretrained(ASPECT_DETECTOR_DIR)
aspect_model     = RobertaForSequenceClassification.from_pretrained(ASPECT_DETECTOR_DIR).to(device)
aspect_model.eval()

with open(f"{ASPECT_DETECTOR_DIR}/thresholds.json") as f:
    THRESHOLDS = json.load(f)

print("Loading sentiment classifier...")
sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_CLASSIFIER_DIR)
sentiment_model     = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_CLASSIFIER_DIR).to(device)
sentiment_model.eval()

print("Models loaded.\n")


def detect_aspects(text: str) -> list[str]:
    """
    Returns a list of aspects detected in the text.
    Uses per-aspect tuned thresholds.
    """
    enc = aspect_tokenizer(
        text,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids      = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)

    with torch.no_grad():
        outputs = aspect_model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.sigmoid(outputs.logits).squeeze(0).cpu().numpy()

    detected = []
    for i, aspect in enumerate(ASPECTS):
        threshold = THRESHOLDS.get(aspect, 0.5)
        if probs[i] >= threshold:
            detected.append(aspect)

    return detected


def classify_sentiment(text: str, aspect: str) -> tuple[str, float]:
    """
    Returns (sentiment_label, confidence) for a given text-aspect pair.
    """
    enc = sentiment_tokenizer(
        text,
        aspect,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids      = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    token_type_ids = enc.get('token_type_ids', torch.zeros(1, MAX_LEN, dtype=torch.long)).to(device)

    with torch.no_grad():
        outputs     = sentiment_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        probs       = torch.softmax(outputs.logits, dim=1).squeeze(0).cpu().numpy()
        pred_class  = int(np.argmax(probs))
        confidence  = float(probs[pred_class])

    return SENTIMENT_MAP[pred_class], confidence


def predict(text: str) -> list[dict]:
    """
    Full pipeline: text → [(aspect, sentiment, confidence), ...]
    Returns an empty list if no aspects are detected.
    """
    aspects = detect_aspects(text)

    if not aspects:
        return []

    results = []
    for aspect in aspects:
        sentiment, confidence = classify_sentiment(text, aspect)
        results.append({
            "aspect"    : aspect,
            "sentiment" : sentiment,
            "confidence": round(confidence, 4)
        })

    return results


def format_output(results: list[dict]) -> str:
    """Pretty print results."""
    if not results:
        return "No aspects detected in this text."
    lines = []
    for r in results:
        lines.append(f"  aspect     → {r['aspect']}")
        lines.append(f"  sentiment  → {r['sentiment']}  (confidence: {r['confidence']:.2%})")
        lines.append("")
    return "\n".join(lines)



if __name__ == "__main__":
    print("ABSA Inference — type a review and press Enter. Type 'quit' to exit.\n")
    while True:
        text = input("Review: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        results = predict(text)
        print()
        print(format_output(results))
