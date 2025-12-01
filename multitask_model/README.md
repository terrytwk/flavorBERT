# Multi-Task Pre-Training: MLM + Food Co-occurrence Prediction

## Experiment Overview

This directory contains a **multi-task pre-training approach** for molecular representation learning. The hypothesis is that jointly training on two objectives will produce better representations for flavor/taste prediction tasks:

1. **Masked Language Modeling (MLM)**: Standard BERT-style pre-training on SMILES strings
2. **Food Co-occurrence Prediction**: Predicting which foods contain a given molecule (multi-label classification)

The idea is that learning food context during pre-training will help the model learn chemically meaningful representations that are specifically useful for flavor-related downstream tasks.

---

## Files

### Core Multi-Task Model

**`model.py`** - Multi-task RoBERTa architecture
- `RobertaMLMAndFoodHead`: Custom model with two heads
  - Shared RoBERTa encoder
  - MLM head for token prediction
  - Food prediction head for multi-label food classification
- Weighted loss combining both objectives

**`dataset.py`** - Multi-task data loader
- `MultiTaskChemDataset`: Combines general chemistry SMILES with FoodDB molecules
- `MultiTaskCollator`: Handles MLM masking and food label batching
- Randomly samples between food-context and general chemistry data

**`train.py`** - Training script
- `MultiTaskTrainer`: Custom trainer with individual loss logging
- Tracks MLM loss and food prediction loss separately
- Command-line interface for configuring multi-task training

---

### Baseline Models

**`vanillaRoberta.py`** - Standard MLM pre-training
- Self-contained script for training vanilla RoBERTa
- No food context, just pure MLM on SMILES
- Used as baseline comparison

---

### Data Preparation Utilities

**`extractBaselineSmiles.py`** - Extract SMILES for baseline training
- Extracts SMILES from FoodDB JSONL
- Optionally merges with additional SMILES (e.g., PubChem sample)
- Creates unified dataset for vanilla/baseline training

**`sample_and_canonicalize.py`** - Sample and canonicalize SMILES
- Randomly samples N molecules from large SMILES files
- Canonicalizes SMILES using RDKit (ensures unique representation)
- Parallelized for efficiency on large datasets

---

## Experiment Design

### Multi-Task Training
```bash
python train.py \
    --general_chem_data baseline_smiles.txt \
    --food_context_data foodb_context.jsonl \
    --food_vocab food_vocab.json \
    --food_task_weight 0.3 \
    --mlm_loss_weight 1.0 \
    --food_loss_weight 1.0
```

### Baseline Training
```bash
python vanillaRoberta.py \
    --dataset_path baseline_smiles.txt \
    --tokenizer_type smiles
```

### Evaluation
After pre-training, both models are evaluated on the FART (Flavor and Aroma Recognition Task) benchmark to compare their performance on taste classification.

---

## Key Idea

**Question**: Does incorporating food context during pre-training improve molecular representations for flavor prediction?

**Approach**: Train a model that simultaneously learns to:
- Reconstruct masked SMILES tokens (general molecular understanding)
- Predict which foods contain each molecule (domain-specific knowledge)

**Hypothesis**: The food prediction task acts as a form of supervised pre-training that biases the model toward chemically meaningful features relevant to taste and flavor.

