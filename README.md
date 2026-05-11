# Turkish LLM Distillation for Multiple-Choice Question Answering

This project investigates a teacher-student distillation pipeline for Turkish multiple-choice question answering using Large Language Models (LLMs), LoRA fine-tuning and ensemble learning techniques.

The study focuses on transferring reasoning behavior from a large teacher model to smaller student models under multiple teacher-generation scenarios.

---

# Project Overview

Large Language Models (LLMs) can achieve strong performance on multiple-choice reasoning tasks, but their computational cost and model size make practical deployment difficult.

This project explores a distillation-based approach where knowledge from a large Turkish reasoning model is transferred to smaller student models using:

* Teacher-Student Distillation
* LoRA Fine-Tuning
* 4-bit Quantization
* Ensemble Learning
* Multiple Teacher Generation Strategies
* Data Augmentation with Choice Permutation

The main objective is to analyze how different teacher labeling strategies affect student model behavior and generalization performance.

---

# Models

## Teacher Model

* `ytu-ce-cosmos/Turkish-Gemma-9b-T1`

## Student Model

* `Metin/gemma-2b-tr-inst`

---

# Methodology

The project pipeline consists of the following stages:

1. Dataset filtering and preprocessing
2. Teacher-generated label creation
3. Format-based output filtering
4. Static choice permutation augmentation
5. LoRA fine-tuning of student models
6. Ensemble evaluation using majority voting

Three different teacher scenarios were evaluated:

* Teacher-1: Greedy generation
* Teacher-2: Sampling-based generation
* Teacher-3: Multi-sampling generation

---

# Training Setup

## Quantization

* 4-bit NF4 Quantization
* Double Quantization
* bf16 computation

## LoRA Configuration

* Rank (r): 16
* Alpha: 32
* Dropout: 0.05

## Training Parameters

* Learning Rate: 2e-4
* Max Length: 384
* Batch Size: 8
* Gradient Accumulation: 2

---

# Dataset

The project uses a filtered subset of the Turkish MMLU dataset:

* Total Questions: 2500
* Training Set: 2000
* Test Set: 500

Dataset Source:
https://zenodo.org/records/16283327

The selected subset focuses on:

* Mathematical reasoning
* Logical reasoning
* Science-oriented questions

---

# Results

## Validation Accuracy (Teacher Labels)

| Model     | Validation Accuracy |
| --------- | ------------------- |
| Student-1 | 0.8158              |
| Student-2 | 0.6111              |
| Student-3 | 0.8182              |

## Test Accuracy (Gold Labels)

| Model     | Test Accuracy |
| --------- | ------------- |
| Student-1 | 0.212         |
| Student-2 | 0.226         |
| Student-3 | 0.222         |
| Ensemble  | 0.214         |

The experiments show that high validation performance against teacher labels does not necessarily translate into strong generalization on gold test labels.

The analysis suggests that:

* teacher-generated label noise,
* low valid formatting ratio,
* and correlated student errors

significantly affect distillation performance.

---

# Repository Structure

```text
results/
├── student1/
├── student2/
├── student3/
└── ensemble/
```

* `DistillEnsemble.ipynb` contains the main experimental workflow.
* `results/` contains training curves and evaluation figures.
* `Rapor.pdf` contains the full project report.

---

# Experimental Outputs

Additional experimental outputs, checkpoints and supplementary files are available via Google Drive:

[Google Drive Link]

---

# Libraries and Tools

* PyTorch
* Hugging Face Transformers
* PEFT (LoRA)
* BitsAndBytes
* Datasets
* Pandas
* NumPy
* Matplotlib
* scikit-learn

---

# Key Topics

* Large Language Models (LLMs)
* Knowledge Distillation
* Ensemble Learning
* LoRA Fine-Tuning
* Quantization
* Turkish NLP
* Multiple-Choice Question Answering

---

# Author

Emir Kaan SAIT

Yıldız Technical University
M.Sc. Computer Engineering
