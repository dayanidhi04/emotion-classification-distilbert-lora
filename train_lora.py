
from pathlib import Path

import numpy as np
import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

import evaluate

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "distilbert-base-uncased"

DATASET_NAME = "dair-ai/emotion"

OUTPUT_DIR = PROJECT_ROOT / "fine_tuned_model"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

NUM_LABELS = 6

LABEL_NAMES = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise",
]

LORA_R = 8

LORA_ALPHA = 16

LORA_DROPOUT = 0.1

LORA_TARGET_MODULES = [
    "q_lin",
    "v_lin",
]

accuracy_metric = evaluate.load("accuracy")

f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):

    logits, labels = eval_pred

    preds = np.argmax(
        logits,
        axis=-1
    )

    acc = accuracy_metric.compute(
        predictions=preds,
        references=labels,
    )

    f1 = f1_metric.compute(
        predictions=preds,
        references=labels,
        average="macro",
    )

    return {
        "accuracy": acc["accuracy"],
        "f1_macro": f1["f1"],
    }

def main():

    print("=" * 60)
    print("DISTILBERT + LoRA EMOTION CLASSIFICATION")
    print("=" * 60)

    device = (
        "CUDA GPU"
        if torch.cuda.is_available()
        else "CPU"
    )

    print(f"\nDevice: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    else:

        print(
            "\nWARNING: CUDA GPU not detected."
        )

        print(
            "Training will run on CPU and may be slow."
        )

    print(
        f"\nLoading dataset: {DATASET_NAME}"
    )

    ds = load_dataset(
        DATASET_NAME
    )

    print("\nDataset loaded.")

    print(
        f"Train: {len(ds['train'])}"
    )

    print(
        f"Validation: {len(ds['validation'])}"
    )

    print(
        f"Test: {len(ds['test'])}"
    )

    print(
        f"\nLoading tokenizer: {MODEL_NAME}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    def tokenize(batch):

        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=128,
        )

    print("\nTokenizing dataset...")

    tokenized = ds.map(
        tokenize,
        batched=True,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    print(
        f"\nLoading base model: {MODEL_NAME}"
    )

    base_model = (
        AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
        )
    )

    print("\nConfiguring LoRA...")

    lora_config = LoraConfig(

        task_type=TaskType.SEQ_CLS,

        r=LORA_R,

        lora_alpha=LORA_ALPHA,

        lora_dropout=LORA_DROPOUT,

        target_modules=LORA_TARGET_MODULES,

        modules_to_save=[
            "pre_classifier",
            "classifier",
        ],
    )

    model = get_peft_model(
        base_model,
        lora_config,
    )

    print("\nLoRA model created.")

    model.print_trainable_parameters()

    training_args = TrainingArguments(

        output_dir=str(
            CHECKPOINT_DIR
        ),

        learning_rate=2e-4,

        per_device_train_batch_size=16,

        per_device_eval_batch_size=32,

        num_train_epochs=5,

        weight_decay=0.01,

        eval_strategy="epoch",

        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="f1_macro",

        greater_is_better=True,

        logging_steps=50,

        report_to="none",

    
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=tokenized["train"],

        eval_dataset=tokenized["validation"],

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics,
    )
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    trainer.train()
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    eval_results = trainer.evaluate()

    for key, value in eval_results.items():

        if isinstance(value, float):

            print(
                f"{key}: {value:.4f}"
            )

        else:

            print(
                f"{key}: {value}"
            )
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    print(
        f"\nSaving LoRA model to:"
    )

    print(
        OUTPUT_DIR
    )

    model.save_pretrained(
        str(OUTPUT_DIR)
    )

    tokenizer.save_pretrained(
        str(OUTPUT_DIR)
    )
    print("\nChecking saved files...")

    adapter_config = (
        OUTPUT_DIR / "adapter_config.json"
    )

    adapter_model = (
        OUTPUT_DIR / "adapter_model.safetensors"
    )

    if adapter_config.exists():

        print(
            "✓ adapter_config.json"
        )

    else:

        print(
            "✗ adapter_config.json NOT FOUND"
        )

    if adapter_model.exists():

        print(
            "✓ adapter_model.safetensors"
        )

    else:

        print(
            "✗ adapter_model.safetensors NOT FOUND"
        )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        "\nNext command:"
    )

    print(
        "python src/evaluate.py"
    )
if __name__ == "__main__":
    main()