# Task 5 — Deep Learning / NLP Text Classifier with PyTorch

## Overview

This task implements a Deep Learning based Natural Language Processing (NLP) pipeline for binary sentiment classification.

The objective is to build a neural network that analyzes text and predicts whether the sentiment of the text is **Positive** or **Negative**.

The project uses **PyTorch** for constructing and training the neural network and **NLTK** for obtaining and processing the sentiment dataset.

---

## Problem Statement

Sentiment analysis is an NLP classification problem where a text input is classified according to its emotional or opinion-based polarity.

For this project, the model performs binary classification:

```text
Text
  ↓
Tokenization
  ↓
Vocabulary / Integer Encoding
  ↓
Padding
  ↓
Embedding Layer
  ↓
Neural Network
  ↓
Positive / Negative
```

Example:

```text
Input:
"I really enjoyed this movie."

Prediction:
Positive

Confidence:
High
```

---

## Dataset

The project uses the **NLTK Movie Reviews** dataset.

The dataset contains movie review documents labeled as either:

* `pos` — Positive
* `neg` — Negative

The dataset
