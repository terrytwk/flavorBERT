"""
Training utilities for FART model.
"""
import os
from typing import Tuple
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder

from model_utils import load_tokenizer, load_model, create_tokenize_function, compute_metrics
from config import ModelConfig, TrainingConfig


def setup_training_args(config: TrainingConfig) -> TrainingArguments:
    """
    Create TrainingArguments from configuration.

    Parameters
    ----------
    config : TrainingConfig
        Training configuration.

    Returns
    -------
    TrainingArguments
        Configured training arguments.
    """
    # Organize output by run_name: results/{run_name}/
    output_dir = os.path.join(config.output_dir, config.run_name) if config.run_name else config.output_dir
    
    return TrainingArguments(
        run_name=config.run_name,
        output_dir=output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        weight_decay=config.weight_decay,
        evaluation_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        load_best_model_at_end=config.load_best_model_at_end,
        save_total_limit=config.save_total_limit,
    )


def create_trainer(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    train_dataset,
    val_dataset,
    test_dataset,
    label_column: str
) -> Tuple[Trainer, Dataset, LabelEncoder]:
    """
    Create and initialize the Trainer.

    Parameters
    ----------
    model_config : ModelConfig
        Model configuration.
    training_config : TrainingConfig
        Training configuration.
    train_dataset : Dataset
        Training dataset.
    val_dataset : Dataset
        Validation dataset.
    test_dataset : Dataset
        Test dataset.
    label_column : str
        Name of the label column.

    Returns
    -------
    tuple
        (trainer, test_dataset, label_encoder)
    """
    # Load tokenizer and model
    tokenizer = load_tokenizer(model_config.tokenizer_checkpoint)
    model = load_model(model_config.model_checkpoint, model_config.num_labels)

    # Create tokenization function
    tokenize_fn = create_tokenize_function(tokenizer, max_length=512)

    # Tokenize all datasets FIRST (matching original code order)
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    val_dataset = val_dataset.map(tokenize_fn, batched=True)
    test_dataset = test_dataset.map(tokenize_fn, batched=True)

    # Label encoding AFTER tokenization (matching original code order)
    # Original code incorrectly used fit_transform on all, but we'll preserve that behavior
    label_encoder = LabelEncoder()
    
    # Original code used fit_transform on all three datasets
    # This preserves the exact original behavior
    encoded_labels_train = label_encoder.fit_transform(train_dataset[label_column])
    train_dataset = train_dataset.add_column('label', encoded_labels_train)

    encoded_labels_val = label_encoder.transform(val_dataset[label_column])
    val_dataset = val_dataset.add_column('label', encoded_labels_val)

    encoded_labels_test = label_encoder.transform(test_dataset[label_column])
    test_dataset = test_dataset.add_column('label', encoded_labels_test)

    # Setup training arguments
    training_args = setup_training_args(training_config)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    return trainer, test_dataset, label_encoder

