"""
Step 3: FART Evaluation Script

This is the complete FART evaluation pipeline from the paper.
To use your custom food-context model from Step 2, you ONLY need to change
ONE line (marked with *** CHANGE THIS LINE ***).

The script handles:
- Data loading and augmentation
- Custom weighted training
- Comprehensive evaluation (confusion matrix, ROC curves, ensemble voting)
"""

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, roc_curve, roc_auc_score, 
                              classification_report)
import evaluate
from torch import nn
import torch
from rdkit import Chem
from collections import Counter
import math
import itertools
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import os


# ============================================================================
# CUSTOM TRAINER CLASS
# ============================================================================

class CustomTrainer(Trainer):
    """
    Custom Trainer class that supports weighted loss for handling class imbalance.
    
    This trainer computes class-weighted cross-entropy loss to give more weight
    to underrepresented classes during training.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute weighted cross-entropy loss.
        
        Args:
            model: The model being trained
            inputs: Dictionary containing input tensors and labels
            return_outputs: Whether to return model outputs along with loss
            
        Returns:
            loss (or tuple of loss and outputs if return_outputs=True)
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Use weighted cross-entropy if class weights are provided
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = nn.CrossEntropyLoss()
            
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ============================================================================
# DATA AUGMENTATION FUNCTIONS
# ============================================================================

def control_smiles_duplication(random_smiles, duplicate_control=lambda x: 1):
    """
    Returns augmented SMILES with the number of duplicates controlled by the function duplicate_control.

    Taken from https://github.com/volkamerlab/maxsmi/blob/main/maxsmi/utils/utils_smiles.py
    """
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
    """
    Takes a SMILES (not necessarily canonical) and returns `int_aug` random variations of this SMILES.

    Taken from https://github.com/volkamerlab/maxsmi/blob/main/maxsmi/utils/utils_smiles.py
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    else:
        if int_aug > 0:
            return [
                Chem.MolToSmiles(mol, canonical=False, doRandom=True)
                for _ in range(int_aug)
            ]
        elif int_aug == 0:
            return [smiles]
        else:
            raise ValueError("int_aug must be greater or equal to zero.")


def augmentation_without_duplication(smiles, augmentation_number):
    """
    Takes a SMILES and returns a list of unique random SMILES.

    Taken from https://github.com/volkamerlab/maxsmi/blob/main/maxsmi/utils/utils_smiles.py
    """
    smiles_list = smiles_to_random(smiles, augmentation_number)
    return control_smiles_duplication(smiles_list, lambda x: 1)


def augment_dataset(dataset, augmentation_numbers, tastes):
    """
    Augments the dataset by generating new SMILES strings for specified tastes.

    Args:
        dataset (Dataset): The original dataset.
        augmentation_numbers (list): Numbers of new SMILES to generate for each taste.
        tastes (list): Taste categories to augment.

    Returns:
        Dataset: Augmented dataset with new SMILES strings.
    """
    augmented_data = []
    for i, taste in enumerate(tastes):
        for entry in dataset:
            if entry["Canonicalized Taste"] == taste:
                original_smiles = entry["Canonicalized SMILES"]
                new_smiles_list = augmentation_without_duplication(original_smiles, augmentation_numbers[i])

                for new_smiles in new_smiles_list:
                    new_entry = deepcopy(entry)
                    new_entry["Canonicalized SMILES"] = new_smiles
                    augmented_data.append(new_entry)
            else:
                augmented_data.append(entry)

    # Convert augmented_data list to Dataset object
    augmented_dataset = Dataset.from_dict({key: [entry[key] for entry in augmented_data] for key in augmented_data[0]})

    return augmented_dataset


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def run_fart_evaluation(
    model_checkpoint="seyonec/SMILES_tokenized_PubChem_shard00_160k",  # *** CHANGE THIS LINE ***
    data_dir="../dataset/splits",
    output_dir="./results",
    run_name="fart_evaluation",
    augmentation=True,
    augmentation_numbers=[10, 10, 10, 10, 10],
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    max_length=512
):
    """
    Run the complete FART evaluation pipeline.
    
    Args:
        model_checkpoint (str): Path to model (e.g., "./my-food-context-model" or HF model ID)
        data_dir (str): Directory containing fart_train.csv, fart_val.csv, fart_test.csv
        output_dir (str): Directory to save training outputs
        run_name (str): Name for the run (for logging)
        augmentation (bool): Whether to perform SMILES augmentation
        augmentation_numbers (list): Number of augmentations per taste category
        num_train_epochs (int): Number of training epochs
        per_device_train_batch_size (int): Training batch size
        per_device_eval_batch_size (int): Evaluation batch size
        max_length (int): Maximum sequence length for tokenization
    """
    
    print("=" * 80)
    print("FART EVALUATION PIPELINE")
    print("=" * 80)
    print(f"Model: {model_checkpoint}")
    print(f"Data directory: {data_dir}")
    print(f"Augmentation: {augmentation}")
    print("=" * 80)
    
    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    print("\n[1/7] Loading data...")
    train_df = pd.read_csv(os.path.join(data_dir, "fart_train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "fart_val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "fart_test.csv"))
    
    # Reset index to avoid "__index_level_0__" column
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    
    # ========================================================================
    # 2. AUGMENTATION
    # ========================================================================
    if augmentation:
        print("\n[2/7] Performing SMILES augmentation...")
        tastes = ['bitter', 'sour', 'sweet', 'umami', 'undefined']
        
        train_dataset = augment_dataset(train_dataset, augmentation_numbers, tastes)
        val_dataset = augment_dataset(val_dataset, augmentation_numbers, tastes)
        test_dataset = augment_dataset(test_dataset, augmentation_numbers, tastes)
        
        print(f"✓ Augmented train samples: {len(train_dataset)}")
        print(f"✓ Augmented validation samples: {len(val_dataset)}")
        print(f"✓ Augmented test samples: {len(test_dataset)}")
    else:
        print("\n[2/7] Skipping augmentation...")
    
    # ========================================================================
    # 3. LOAD MODEL AND TOKENIZER
    # ========================================================================
    print(f"\n[3/7] Loading model and tokenizer from: {model_checkpoint}")
    
    # The tokenizer will automatically load from the same directory
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    print(f"✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
    
    # ========================================================================
    # 4. TOKENIZATION
    # ========================================================================
    print("\n[4/7] Tokenizing datasets...")
    
    def tokenize_function(examples):
        return tokenizer(examples["Canonicalized SMILES"], padding="max_length", 
                        truncation=True, max_length=max_length)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    print("✓ Tokenization complete")
    
    # ========================================================================
    # 5. LABEL ENCODING
    # ========================================================================
    print("\n[5/7] Encoding labels...")
    label_encoder = LabelEncoder()
    
    encoded_labels = label_encoder.fit_transform(train_dataset['Canonicalized Taste'])
    train_dataset = train_dataset.add_column('label', encoded_labels)
    
    encoded_labels = label_encoder.transform(val_dataset['Canonicalized Taste'])
    val_dataset = val_dataset.add_column('label', encoded_labels)
    
    encoded_labels = label_encoder.transform(test_dataset['Canonicalized Taste'])
    test_dataset = test_dataset.add_column('label', encoded_labels)
    
    print("✓ Label encoding complete")
    print(f"✓ Classes: {label_encoder.classes_}")
    
    # Compute class weights for handling class imbalance
    train_labels = np.array(train_dataset['label'])
    class_weight_values = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weight_values, dtype=torch.float32)
    
    print("\nClass distribution in training set:")
    unique, counts = np.unique(train_labels, return_counts=True)
    for label, count in zip(unique, counts):
        class_name = label_encoder.inverse_transform([label])[0]
        print(f"  {class_name}: {count} samples (weight: {class_weight_values[label]:.4f})")
    
    # ========================================================================
    # 6. TRAINING
    # ========================================================================
    print("\n[6/7] Setting up training...")
    
    num_labels = 5
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, 
        num_labels=num_labels
    )
    print(f"✓ Classification head initialized with {num_labels} labels")
    
    metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    training_args = TrainingArguments(
        run_name=run_name,
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        weight_decay=0.01,
        eval_strategy="steps",
        logging_dir=os.path.join(output_dir, "logs"),
        save_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=5,
        # DataLoader optimizations (doesn't change experimental design)
        dataloader_num_workers=16,     # 16 workers for 32 CPUs (8 per GPU)
        dataloader_pin_memory=True,    # Faster CPU->GPU transfer
        dataloader_prefetch_factor=2,  # Prefetch 2 batches per worker
    )
    
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
    
    # ========================================================================
    # 7. EVALUATION
    # ========================================================================
    print("\n[7/7] Running evaluation...")
    
    # Evaluate on validation set
    print("\nValidation Results:")
    results = trainer.evaluate(eval_dataset=val_dataset)
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    # Predict on test set
    print("\nGenerating test predictions...")
    predictions = trainer.predict(test_dataset)
    
    probs = softmax(predictions.predictions, axis=1)
    pred_labels = np.argmax(probs, axis=1)
    true_labels = predictions.label_ids
    
    # Confusion Matrix
    print("\nGenerating confusion matrix...")
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    label_names = label_encoder.inverse_transform(range(5))
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {output_dir}/confusion_matrix.png")
    
    # Test Metrics
    print("\nTest Set Results:")
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"  Accuracy: {accuracy:.4f}")
    
    precision, recall, f1, support = precision_recall_fscore_support(true_labels, pred_labels)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='macro'
    )
    
    print(f"  Macro Precision: {precision_weighted:.4f}")
    print(f"  Macro Recall: {recall_weighted:.4f}")
    print(f"  Macro F1 Score: {f1_weighted:.4f}")
    
    print("\nPer-Class Metrics:")
    class_names = label_encoder.classes_
    for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
        print(f"  Class {class_names[i]}:")
        print(f"    Precision: {p:.4f}")
        print(f"    Recall: {r:.4f}")
        print(f"    F1 Score: {f:.4f}")
        print(f"    Support: {s}")
    
    # ROC Curves
    print("\nGenerating ROC curves...")
    num_classes = probs.shape[1]
    true_labels_bin = label_binarize(true_labels, classes=np.arange(num_classes))
    
    plt.figure(figsize=(10, 8))
    auc_scores = []
    for i in range(num_classes):
        if np.sum(true_labels_bin[:, i]) > 0:
            auc = roc_auc_score(true_labels_bin[:, i], probs[:, i])
            auc_scores.append(auc)
            fpr, tpr, _ = roc_curve(true_labels_bin[:, i], probs[:, i])
            plt.plot(fpr, tpr, label=f"{label_names[i]} (AUC = {auc:.4f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Each Class")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    print(f"✓ ROC curves saved to {output_dir}/roc_curves.png")
    
    # Ensemble Voting (if augmentation was used)
    if augmentation:
        print("\nPerforming ensemble voting...")
        df = pd.DataFrame({
            'Standardized SMILES': test_dataset['Standardized SMILES'],
            'label': test_dataset['label'],
            'pred_probs': list(probs)
        })
        df['pred_labels'] = np.argmax(df['pred_probs'].tolist(), axis=1)
        
        # Majority voting for each molecule
        grouped = df.groupby('Standardized SMILES').agg({
            'pred_labels': lambda x: x.value_counts().idxmax() if x.value_counts().iloc[0] >= 10 else np.nan,
            'label': 'first'
        }).dropna().reset_index()
        
        # Average probabilities
        pred_probs_voted = np.array([
            np.mean(df.loc[df['Standardized SMILES'] == smile, 'pred_probs'], axis=0)
            for smile in grouped['Standardized SMILES']
        ])
        
        true_labels_voted = grouped['label'].values
        pred_labels_voted = grouped['pred_labels'].values
        
        # Voted metrics
        accuracy_voted = accuracy_score(true_labels_voted, pred_labels_voted)
        print(f"  Voted Accuracy: {accuracy_voted:.4f}")
        
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels_voted, pred_labels_voted, labels=np.arange(num_classes)
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            true_labels_voted, pred_labels_voted, average='macro', labels=np.arange(num_classes)
        )
        
        print(f"  Voted Macro Precision: {precision_weighted:.4f}")
        print(f"  Voted Macro Recall: {recall_weighted:.4f}")
        print(f"  Voted Macro F1 Score: {f1_weighted:.4f}")
        
        # Voted ROC curves
        true_labels_bin_voted = label_binarize(true_labels_voted, classes=np.arange(num_classes))
        
        plt.figure(figsize=(10, 8))
        for i in range(num_classes):
            if np.sum(true_labels_bin_voted[:, i]) > 0:
                auc = roc_auc_score(true_labels_bin_voted[:, i], pred_probs_voted[:, i])
                fpr, tpr, _ = roc_curve(true_labels_bin_voted[:, i], pred_probs_voted[:, i])
                plt.plot(fpr, tpr, label=f"{label_names[i]} (AUC = {auc:.4f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.5)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves for Each Class (Ensemble Voting)")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, 'roc_curves_voted.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Ensemble ROC curves saved to {output_dir}/roc_curves_voted.png")
    
    print("\n" + "=" * 80)
    print("✓ EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    
    return trainer, results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run FART evaluation pipeline")
    
    # *** THIS IS THE KEY ARGUMENT TO CHANGE ***
    parser.add_argument("--model_checkpoint", type=str, 
                        default="seyonec/SMILES_tokenized_PubChem_shard00_160k",
                        help="Path to model (use './my-food-context-model' for your trained model)")
    
    parser.add_argument("--data_dir", type=str, 
                        default="../dataset/splits",
                        help="Directory containing FART train/val/test CSV files")
    parser.add_argument("--output_dir", type=str, 
                        default="./results",
                        help="Directory to save results")
    parser.add_argument("--run_name", type=str, 
                        default="fart_evaluation",
                        help="Name for the run")
    parser.add_argument("--no_augmentation", action="store_true",
                        help="Disable SMILES augmentation")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and evaluation")
    
    args = parser.parse_args()
    
    # Run evaluation
    run_fart_evaluation(
        model_checkpoint=args.model_checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        augmentation=not args.no_augmentation,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size
    )