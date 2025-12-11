"""
Standalone FART evaluation/training script (mirrors the notebook pipeline).

This is a port of the multitask_model/fart_evaluate.py flow into the
fart/models package so it works with the configs/utilities here. It trains a
Roberta classifier, evaluates on val/test, writes plots, and stores a
run_summary.json in the chosen output directory.
"""

import argparse
import itertools
import json
import math
import os
from collections import Counter
from copy import deepcopy
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset
from rdkit import Chem
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


# ============================================================================
# Helper functions
# ============================================================================

def control_smiles_duplication(random_smiles, duplicate_control=lambda x: 1):
    counted_smiles = Counter(random_smiles)
    smiles_duplication = {
        smiles: math.ceil(duplicate_control(counted_smiles[smiles]))
        for smiles in counted_smiles
    }
    return list(
        itertools.chain.from_iterable(
            [[smiles] * smiles_duplication[smiles] for smiles in smiles_duplication]
        )
    )


def smiles_to_random(smiles, int_aug=50):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if int_aug > 0:
        return [
            Chem.MolToSmiles(mol, canonical=False, doRandom=True)
            for _ in range(int_aug)
        ]
    if int_aug == 0:
        return [smiles]
    raise ValueError("int_aug must be greater or equal to zero.")


def augmentation_without_duplication(smiles, augmentation_number):
    smiles_list = smiles_to_random(smiles, augmentation_number)
    return control_smiles_duplication(smiles_list, lambda x: 1)


def augment_dataset(dataset: Dataset, augmentation_numbers, tastes, label_column, smiles_column):
    augmented_data = []
    for i, taste in enumerate(tastes):
        for entry in dataset:
            if entry[label_column] == taste:
                original_smiles = entry[smiles_column]
                new_smiles_list = augmentation_without_duplication(original_smiles, augmentation_numbers[i])
                for new_smiles in new_smiles_list:
                    new_entry = deepcopy(entry)
                    new_entry[smiles_column] = new_smiles
                    augmented_data.append(new_entry)
            else:
                augmented_data.append(entry)
    return Dataset.from_dict({key: [entry[key] for entry in augmented_data] for key in augmented_data[0]})


class CustomTrainer(Trainer):
    """Trainer with optional class-weighted loss."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def load_csvs(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(os.path.join(data_dir, "fart_train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "fart_val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "fart_test.csv"))
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    return train_df, val_df, test_df


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_run_summary(trainer: Trainer, eval_results: Dict, test_results: Dict, output_dir: str, run_name: str):
    """Save a wandb-like run summary alongside generated artifacts."""
    ensure_dir(output_dir)
    state = trainer.state

    # Collect recent metrics from trainer log history
    all_metrics = {}
    if hasattr(state, "log_history") and state.log_history:
        for entry in state.log_history:
            for key, value in entry.items():
                all_metrics[key] = value

    summary = {
        "run_name": run_name,
        "output_dir": output_dir,
        "eval": eval_results,
        "test": test_results,
        "train": {
            k.replace("train_", "") if k.startswith("train_") else k: v
            for k, v in all_metrics.items()
            if k.startswith("train_") or k in ["loss", "learning_rate", "grad_norm", "epoch", "step"]
        },
    }

    if hasattr(state, "global_step"):
        summary["train"]["global_step"] = state.global_step
    if hasattr(state, "total_flos"):
        summary["total_flos"] = state.total_flos
    if hasattr(state, "best_metric"):
        summary["best_metric"] = state.best_metric
    if hasattr(state, "best_model_checkpoint"):
        summary["best_model_checkpoint"] = state.best_model_checkpoint

    summary_path = os.path.join(output_dir, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary_path


# ============================================================================
# Main pipeline
# ============================================================================

def run_fart_evaluation(
    model_checkpoint="seyonec/SMILES_tokenized_PubChem_shard00_160k",
    data_dir="fart/dataset/splits",
    output_dir="./fart/models/results",
    run_name="fart_evaluation",
    augmentation=False,
    augmentation_numbers=None,
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    max_length=512,
):
    if augmentation_numbers is None:
        augmentation_numbers = [10, 10, 10, 10, 10]
    tastes = ["bitter", "sour", "sweet", "umami", "undefined"]
    label_column = "Canonicalized Taste"
    smiles_column = "Canonicalized SMILES"

    final_output_dir = ensure_dir(os.path.join(output_dir, run_name))

    print("=" * 80)
    print("FART EVALUATION PIPELINE")
    print("=" * 80)
    print(f"Model: {model_checkpoint}")
    print(f"Data directory: {data_dir}")
    print(f"Augmentation: {augmentation}")
    print(f"Output directory: {final_output_dir}")
    print("=" * 80)

    # 1. Load data
    print("\n[1/7] Loading data...")
    train_df, val_df, test_df = load_csvs(data_dir)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")

    # 2. Augmentation
    if augmentation:
        print("\n[2/7] Performing SMILES augmentation...")
        train_dataset = augment_dataset(train_dataset, augmentation_numbers, tastes, label_column, smiles_column)
        val_dataset = augment_dataset(val_dataset, augmentation_numbers, tastes, label_column, smiles_column)
        test_dataset = augment_dataset(test_dataset, augmentation_numbers, tastes, label_column, smiles_column)
        print(f"✓ Augmented train samples: {len(train_dataset)}")
        print(f"✓ Augmented validation samples: {len(val_dataset)}")
        print(f"✓ Augmented test samples: {len(test_dataset)}")
    else:
        print("\n[2/7] Skipping augmentation...")

    # 3. Load tokenizer
    print(f"\n[3/7] Loading model and tokenizer from: seyonec/SMILES_tokenized_PubChem_shard00_160k")
    tokenizer = AutoTokenizer.from_pretrained("seyonec/SMILES_tokenized_PubChem_shard00_160k")
    print(f"✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")

    # 4. Tokenization
    print("\n[4/7] Tokenizing datasets...")

    def tokenize_function(examples):
        return tokenizer(
            examples[smiles_column],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    print("✓ Tokenization complete")

    # 5. Label encoding
    print("\n[5/7] Encoding labels...")
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_dataset[label_column])
    val_labels = label_encoder.transform(val_dataset[label_column])
    test_labels = label_encoder.transform(test_dataset[label_column])

    train_dataset = train_dataset.add_column("labels", train_labels)
    val_dataset = val_dataset.add_column("labels", val_labels)
    test_dataset = test_dataset.add_column("labels", test_labels)
    print(f"✓ Classes: {label_encoder.classes_}")

    # Class weights (disabled to mirror legacy behavior)
    class_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels,
    )
    # class_weights = torch.tensor(class_weight_values, dtype=torch.float32)
    class_weights = None
    print("\nClass distribution in training set:")
    unique, counts = np.unique(train_labels, return_counts=True)
    for label, count in zip(unique, counts):
        class_name = label_encoder.inverse_transform([label])[0]
        print(f"  {class_name}: {count} samples (weight: {class_weight_values[label]:.4f})")

    # 6. Training setup
    print("\n[6/7] Setting up training...")
    num_labels = len(label_encoder.classes_)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
    )
    print(f"✓ Classification head initialized with {num_labels} labels")

    training_args = TrainingArguments(
        run_name=run_name,
        output_dir=final_output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        weight_decay=0.01,
        evaluation_strategy="steps",
        logging_dir=os.path.join(final_output_dir, "logs"),
        save_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=5,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, predictions)}

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    print("\nStarting training...")
    trainer.train()
    print("✓ Training complete!")

    # 7. Evaluation
    print("\n[7/7] Running evaluation...")
    print("\nValidation Results:")
    val_results = trainer.evaluate(eval_dataset=val_dataset)
    for key, value in val_results.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\nGenerating test predictions...")
    predictions = trainer.predict(test_dataset)
    probs = softmax(predictions.predictions, axis=1)
    pred_labels = np.argmax(probs, axis=1)
    true_labels = predictions.label_ids

    # Confusion matrix
    print("\nGenerating confusion matrix...")
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    label_names = label_encoder.inverse_transform(range(num_labels))

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    confusion_path = os.path.join(final_output_dir, "confusion_matrix.png")
    plt.savefig(confusion_path, dpi=300, bbox_inches="tight")
    print(f"✓ Confusion matrix saved to {confusion_path}")
    plt.close()

    # Test metrics
    print("\nTest Set Results:")
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, labels=np.arange(num_labels)
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="macro", labels=np.arange(num_labels)
    )
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro Precision: {precision_weighted:.4f}")
    print(f"  Macro Recall: {recall_weighted:.4f}")
    print(f"  Macro F1 Score: {f1_weighted:.4f}")
    print("\nPer-Class Metrics:")
    for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
        print(f"  Class {label_names[i]}:")
        print(f"    Precision: {p:.4f}")
        print(f"    Recall: {r:.4f}")
        print(f"    F1 Score: {f:.4f}")
        print(f"    Support: {s}")

    # ROC curves
    print("\nGenerating ROC curves...")
    true_labels_bin = label_binarize(true_labels, classes=np.arange(num_labels))
    plt.figure(figsize=(10, 8))
    for i in range(num_labels):
        if np.sum(true_labels_bin[:, i]) > 0:
            auc = roc_auc_score(true_labels_bin[:, i], probs[:, i])
            fpr, tpr, _ = roc_curve(true_labels_bin[:, i], probs[:, i])
            plt.plot(fpr, tpr, label=f"{label_names[i]} (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Each Class")
    plt.legend(loc="lower right")
    roc_path = os.path.join(final_output_dir, "roc_curves.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    print(f"✓ ROC curves saved to {roc_path}")
    plt.close()

    # Ensemble voting when augmented
    ensemble_paths = {}
    if augmentation:
        print("\nPerforming ensemble voting...")
        df = pd.DataFrame(
            {
                "Standardized SMILES": test_dataset["Standardized SMILES"],
                "label": test_dataset["labels"],
                "pred_probs": list(probs),
            }
        )
        df["pred_labels"] = np.argmax(df["pred_probs"].tolist(), axis=1)

        grouped = (
            df.groupby("Standardized SMILES")
            .agg(
                {
                    "pred_labels": lambda x: x.value_counts().idxmax()
                    if x.value_counts().iloc[0] >= 10
                    else np.nan,
                    "label": "first",
                }
            )
            .dropna()
            .reset_index()
        )

        pred_probs_voted = np.array(
            [np.mean(df.loc[df["Standardized SMILES"] == smile, "pred_probs"], axis=0) for smile in grouped["Standardized SMILES"]]
        )
        true_labels_voted = grouped["label"].values
        pred_labels_voted = grouped["pred_labels"].values

        accuracy_voted = accuracy_score(true_labels_voted, pred_labels_voted)
        precision_v, recall_v, f1_v, support_v = precision_recall_fscore_support(
            true_labels_voted, pred_labels_voted, labels=np.arange(num_labels)
        )
        precision_weighted_v, recall_weighted_v, f1_weighted_v, _ = precision_recall_fscore_support(
            true_labels_voted, pred_labels_voted, average="macro", labels=np.arange(num_labels)
        )

        print(f"  Voted Accuracy: {accuracy_voted:.4f}")
        print(f"  Voted Macro Precision: {precision_weighted_v:.4f}")
        print(f"  Voted Macro Recall: {recall_weighted_v:.4f}")
        print(f"  Voted Macro F1 Score: {f1_weighted_v:.4f}")

        true_labels_bin_voted = label_binarize(true_labels_voted, classes=np.arange(num_labels))
        plt.figure(figsize=(10, 8))
        for i in range(num_labels):
            if np.sum(true_labels_bin_voted[:, i]) > 0:
                auc = roc_auc_score(true_labels_bin_voted[:, i], pred_probs_voted[:, i])
                fpr, tpr, _ = roc_curve(true_labels_bin_voted[:, i], pred_probs_voted[:, i])
                plt.plot(fpr, tpr, label=f"{label_names[i]} (AUC = {auc:.4f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.5)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves for Each Class (Ensemble Voting)")
        plt.legend(loc="lower right")
        roc_voted_path = os.path.join(final_output_dir, "roc_curves_voted.png")
        plt.savefig(roc_voted_path, dpi=300, bbox_inches="tight")
        print(f"✓ Ensemble ROC curves saved to {roc_voted_path}")
        plt.close()
        ensemble_paths["roc_curves_voted"] = roc_voted_path

    # Persist run summary with metrics and artifact locations
    test_results = {
        "accuracy": accuracy,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "per_class_metrics": {
            label_names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(num_labels)
        },
    }
    eval_results_clean = {k.replace("eval_", ""): v for k, v in val_results.items() if k.startswith("eval_")}
    artifact_paths = {"confusion_matrix": confusion_path, "roc_curves": roc_path, **ensemble_paths}
    summary_path = save_run_summary(
        trainer,
        {**eval_results_clean, "raw": val_results},
        {**test_results, "artifacts": artifact_paths},
        final_output_dir,
        run_name,
    )
    print(f"\n✓ Run summary saved to {summary_path}")

    print("\n" + "=" * 80)
    print("✓ EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {final_output_dir}")

    return trainer, val_results, test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FART evaluation pipeline")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="seyonec/SMILES_tokenized_PubChem_shard00_160k",
        help="Path to model checkpoint or HF model id",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="fart/dataset/splits",
        help="Directory containing fart_train.csv, fart_val.csv, fart_test.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./fart/models/results",
        help="Base directory to save results (run_name will be appended)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="fart_evaluation",
        help="Name for this run (used to namespace output_dir)",
    )
    parser.add_argument(
        "--no_augmentation",
        action="store_true",
        help="Disable SMILES augmentation",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and evaluation",
    )
    args = parser.parse_args()

    run_fart_evaluation(
        model_checkpoint=args.model_checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        augmentation=not args.no_augmentation,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
    )
