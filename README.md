# CoAttentionNLI

🧠 **SBERT-Guided Co-Attention Model for Natural Language Inference with Curriculum Learning**

This repository presents a neural inference system that captures **semantic relationships between sentence pairs** using a hybrid attention-based architecture. By combining **DeBERTa**, **Co-Attention**, **SBERT embeddings**, and **Curriculum Learning**, the model learns to classify sentence-level relations (entailment, neutral, contradiction) with high interpretability and accuracy.

The system leverages both token-level and sentence-level semantics:
- **DeBERTa** for deep contextual embeddings,
- **SBERT** for semantic sentence similarity,
- **Cross-level fusion** for enhanced representation,
- **Prototypical contrastive loss** for improved class separation.

---

## 🔍 Overview

- **Premise ↔ Hypothesis** pairs are processed for semantic alignment.
- **Modality A:** DeBERTa-v3-base (token-level)
- **Modality B:** SBERT (sentence-level, static)
- **Fusion Strategy:** Cross-modality fusion with Co-Attention and Classwise Pooling.
- **Training Strategy:** Curriculum Learning (easy → medium → hard) + Multi-loss (CE + contrastive + similarity).

---

## 🔧 Architecture

              Premise & Hypothesis Pair
                        ↓
          ┌─────────────────────────────┐
          │     Tokenizer (DeBERTa)     │
          └─────────────────────────────┘
                        ↓
            Contextual Token Embeddings
                        ↓
         [SEP]-Based Token Splitting (P / H)
                        ↓
        Co-Attention: Premise <--> Hypothesis
                        ↓
           Multi-Head Attention Blocks
                        ↓
          Class-wise Attention Pooling
                 ↓              ↓
          Premise_repr     Hypothesis_repr
                 ↓              ↓
          SBERT(Premise)    SBERT(Hypothesis)
                 ↓              ↓
           🔗 Concatenation & Fusion
                        ↓
            Projection Layer + Dropout
                        ↓
      ┌─────────────────────────────────────┐
      │   Classifier Head (CE + Contrast)   │
      └─────────────────────────────────────┘
                        ↓
              Final Prediction (3 classes)
---

**📂 Datasets Used**
Dataset	Description
SNLI	Pretraining & curriculum start
MNLI	Fine-tuning with class difficulty
Matched	Standard validation evaluation
Mismatched	Robustness test with domain shift

Each dataset is pre-annotated with:

SBERT embeddings (premise_sbert, hypothesis_sbert)

Difficulty labels: easy, medium, hard

Similarity scores (SBERT cosine)

---

**🧪 Training Summary**
Mixed Precision (AMP)

Optimizer: Lookahead + AdamW

LR Scheduler: ReduceLROnPlateau

Epochs: 10 (SNLI), 25 (MNLI)

Batch Size: 64

Early Stopping with patience = 2

---


**📈 Evaluation Features**
classification_report (precision, recall, F1)

Confusion Matrix (Seaborn)

High-Confidence Mistake Logging

Loss Histogram for instance-level error inspection

Most-Confused Class Sample Display
