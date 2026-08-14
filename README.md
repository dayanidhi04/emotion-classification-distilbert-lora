# 😊 Emotion Classification using DistilBERT + LoRA

A transformer-based NLP project for classifying text into **6 emotion categories** using **DistilBERT fine-tuned with LoRA (Low-Rank Adaptation)**.

The project includes model fine-tuning, evaluation, and a Gradio interface for real-time emotion prediction.

---

## 🚀 Demo

The application accepts a text sentence and predicts the emotion with confidence scores.

### Example

**Input:**

> I am extremely happy today!

**Prediction:**

> 😊 Joy

---

## 🎯 Objective

The goal of this project is to build an efficient emotion classification model while reducing the number of trainable parameters during fine-tuning.

Instead of fully fine-tuning DistilBERT, **LoRA adapters** are used to efficiently adapt the pretrained model to the emotion classification task.

---

## 🧠 Model

### Base Model

```text
distilbert-base-uncased
