
import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

MODEL_NAME = "distilbert-base-uncased"
ADAPTER_DIR = "fine_tuned_model"
DATASET_NAME = "dair-ai/emotion"
LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
BASELINE_RESULTS_PATH = "results/baseline_results.json"
OUTPUT_DIR = "results"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading test set...")
    ds = load_dataset(DATASET_NAME)
    test = ds["test"]

    print("Loading fine-tuned (base + LoRA adapter) model...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_NAMES)
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    preds, labels = [], []
    with torch.no_grad():
        for example in test:
            inputs = tokenizer(
                example["text"], return_tensors="pt", truncation=True, max_length=128
            ).to(device)
            logits = model(**inputs).logits
            pred = int(torch.argmax(logits, dim=-1))
            preds.append(pred)
            labels.append(example["label"])

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    report = classification_report(
        labels, preds, target_names=LABEL_NAMES, output_dict=True
    )

    print(f"\nFine-tuned model -- Accuracy: {acc:.4f} | Macro F1: {f1_macro:.4f}")

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Fine-tuned Model -- Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
    print(f"Saved confusion matrix to {OUTPUT_DIR}/confusion_matrix.png")

    finetuned_results = {
        "model": "distilbert-base-uncased + LoRA (fine-tuned)",
        "n_samples": len(test),
        "accuracy": acc,
        "f1_macro": f1_macro,
        "classification_report": report,
    }
    with open(f"{OUTPUT_DIR}/finetuned_results.json", "w") as f:
        json.dump(finetuned_results, f, indent=2)

    if os.path.exists(BASELINE_RESULTS_PATH):
        with open(BASELINE_RESULTS_PATH) as f:
            baseline = json.load(f)

        table = (
            "| Model | Accuracy | Macro F1 |\n"
            "|---|---|---|\n"
            f"| Zero-shot baseline (bart-large-mnli) | "
            f"{baseline['accuracy']:.3f} | {baseline['f1_macro']:.3f} |\n"
            f"| Fine-tuned (DistilBERT + LoRA) | {acc:.3f} | {f1_macro:.3f} |\n"
        )
        with open(f"{OUTPUT_DIR}/comparison_table.md", "w") as f:
            f.write(table)
        print(f"\nComparison table saved to {OUTPUT_DIR}/comparison_table.md:\n")
        print(table)
    else:
        print(
            f"\nNo baseline results found at {BASELINE_RESULTS_PATH}. "
            "Run src/baseline_eval.py first to generate the comparison table."
        )


if __name__ == "__main__":
    main()