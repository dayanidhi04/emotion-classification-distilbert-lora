import torch
import gradio as gr

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_SOURCE = "rgmanoj11/emotion-distilbert-lora"
BASE_MODEL = "distilbert-base-uncased"

LABEL_NAMES = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
]


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

print("=" * 60)
print("Emotion Classifier — Local CPU")
print("=" * 60)
print("Using device:", device)


# ============================================================
# TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_SOURCE
)


# ============================================================
# BASE MODEL
# ============================================================

print("Loading DistilBERT...")

base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(LABEL_NAMES)
)


# ============================================================
# LoRA ADAPTER
# ============================================================

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    MODEL_SOURCE
)

model = model.to(device)

model.eval()

print("Model loaded successfully!")


# ============================================================
# PREDICTION
# ============================================================

def predict(text):

    if not text or not text.strip():
        return {}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    probabilities = probabilities.cpu().numpy()

    return {
        LABEL_NAMES[i]: float(probabilities[i])
        for i in range(len(LABEL_NAMES))
    }


# ============================================================
# GRADIO
# ============================================================

demo = gr.Interface(
    fn=predict,

    inputs=gr.Textbox(
        lines=4,
        label="Enter text",
        placeholder="Type a sentence..."
    ),

    outputs=gr.Label(
        num_top_classes=6,
        label="Emotion Prediction"
    ),

    title="Emotion Classifier — DistilBERT + LoRA",

    description=(
        "Fine-tuned DistilBERT using LoRA "
        "for 6-class emotion classification."
    ),

    examples=[
        ["I am extremely happy today!"],
        ["I am scared about what will happen."],
        ["I love spending time with my family."],
        ["This situation makes me very angry."],
        ["I feel really sad today."],
        ["Wow! I never expected that!"]
    ]
)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print("Starting Gradio application...")

    demo.launch()