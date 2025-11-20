"""
Model and tokenizer utilities for FART model.
"""
import numpy as np
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load metric once at module level (more efficient, matches original code)
_accuracy_metric = evaluate.load("accuracy")


def load_tokenizer(tokenizer_checkpoint: str):
    """
    Load tokenizer from checkpoint.

    Parameters
    ----------
    tokenizer_checkpoint : str
        Path or identifier to the tokenizer checkpoint.

    Returns
    -------
    AutoTokenizer
        Loaded tokenizer.
    """
    return AutoTokenizer.from_pretrained(tokenizer_checkpoint)


def load_model(model_checkpoint: str, num_labels: int):
    """
    Load model from checkpoint.

    Parameters
    ----------
    model_checkpoint : str
        Path or identifier to the model checkpoint.
    num_labels : int
        Number of classification labels.

    Returns
    -------
    AutoModelForSequenceClassification
        Loaded model.
    """
    return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)


def create_tokenize_function(tokenizer, max_length: int = 512):
    """
    Create a tokenization function for use with dataset.map().

    Parameters
    ----------
    tokenizer : AutoTokenizer
        Tokenizer to use for tokenization.
    max_length : int
        Maximum sequence length.

    Returns
    -------
    callable
        Tokenization function that takes examples and returns tokenized examples.
    """
    def tokenize_function(examples):
        smiles_column = "Canonicalized SMILES"
        return tokenizer(
            examples[smiles_column],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    return tokenize_function


def compute_metrics(eval_pred):
    """
    Computes accuracy metrics for evaluation predictions.

    Parameters
    ----------
    eval_pred : tuple
        A tuple containing logits and labels.

    Returns
    -------
    dict
        A dictionary with the accuracy metric.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return _accuracy_metric.compute(predictions=predictions, references=labels)

