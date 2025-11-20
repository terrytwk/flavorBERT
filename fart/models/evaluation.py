"""
Evaluation utilities for FART model.
"""
import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder


def evaluate_model(trainer, test_dataset, label_encoder: LabelEncoder):
    """
    Evaluate model on test dataset and compute metrics.

    Parameters
    ----------
    trainer : Trainer
        Trained trainer instance.
    test_dataset : Dataset
        Test dataset.
    label_encoder : LabelEncoder
        Fitted label encoder.

    Returns
    -------
    dict
        Dictionary containing evaluation results with keys:
        - accuracy
        - precision_weighted
        - recall_weighted
        - f1_weighted
        - per_class_metrics (dict mapping class names to metrics)
        - predictions
        - probabilities
        - true_labels
    """
    # Get predictions
    predictions = trainer.predict(test_dataset)
    
    # Compute probabilities and predicted labels
    probs = softmax(predictions.predictions, axis=1)
    pred_labels = np.argmax(probs, axis=1)
    true_labels = predictions.label_ids

    # Calculate overall accuracy
    accuracy = accuracy_score(true_labels, pred_labels)

    # Calculate precision, recall, and F1 score for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, labels=np.arange(len(label_encoder.classes_))
    )

    # Calculate macro-averaged scores
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='macro', labels=np.arange(len(label_encoder.classes_))
    )

    # Get class names
    class_names = label_encoder.classes_

    # Create per-class metrics dictionary
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        }

    return {
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'per_class_metrics': per_class_metrics,
        'predictions': pred_labels,
        'probabilities': probs,
        'true_labels': true_labels
    }


def print_evaluation_results(results: dict, label_encoder: LabelEncoder):
    """
    Print evaluation results in a formatted way.

    Parameters
    ----------
    results : dict
        Results dictionary from evaluate_model().
    label_encoder : LabelEncoder
        Fitted label encoder.
    """
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Weighted Precision: {results['precision_weighted']:.4f}")
    print(f"Weighted Recall: {results['recall_weighted']:.4f}")
    print(f"Weighted F1 Score: {results['f1_weighted']:.4f}")

    print("\nPer-class metrics:")
    for class_name, metrics in results['per_class_metrics'].items():
        print(f"Class {class_name}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  Support: {metrics['support']}")

