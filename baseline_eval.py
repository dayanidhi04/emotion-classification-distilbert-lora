
import json
import random

from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATASET_NAME = "dair-ai/emotion"
LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
N_TEST_SAMPLES = 500  
SEED = 42
OUTPUT_PATH = "results/baseline_results.json"


def main():
    random.seed(SEED)

    print(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)
    test = ds["test"]

    if N_TEST_SAMPLES and N_TEST_SAMPLES < len(test):
        idx = random.sample(range(len(test)), N_TEST_SAMPLES)
        test = test.select(idx)

    print(f"Evaluating zero-shot baseline on {len(test)} test examples...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0,  
    )

    preds, labels = [], []
    for i, example in enumerate(test):
        result = classifier(example["text"], candidate_labels=LABEL_NAMES)
        pred_label = result["labels"][0]
        preds.append(LABEL_NAMES.index(pred_label))
        labels.append(example["label"])

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(test)} done")

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    report = classification_report(
        labels, preds, target_names=LABEL_NAMES, output_dict=True
    )

    print(f"\nZero-shot baseline -- Accuracy: {acc:.4f} | Macro F1: {f1_macro:.4f}")

    import os
    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            {
                "model": "facebook/bart-large-mnli (zero-shot, no fine-tuning)",
                "n_samples": len(test),
                "accuracy": acc,
                "f1_macro": f1_macro,
                "classification_report": report,
            },
            f,
            indent=2,
        )
    print(f"Saved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()