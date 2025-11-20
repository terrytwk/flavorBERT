"""
Main training and evaluation script for FART (Flavor and Aroma Recognition Task) model.

This script orchestrates the complete pipeline:
1. Data loading and preprocessing
2. Optional data augmentation
3. Model training
4. Model evaluation

Usage:
    python FART_Models.py [--run_name RUN_NAME] [--model_checkpoint MODEL_CHECKPOINT]
"""
import argparse
import json
import logging
import os
import sys
from typing import Tuple
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder

from config import ModelConfig, DataConfig, AugmentationConfig, TrainingConfig
from data_utils import load_datasets, augment_dataset
from train import create_trainer
from evaluation import evaluate_model, print_evaluation_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def prepare_datasets(
    data_config: DataConfig,
    augmentation_config: AugmentationConfig
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load and prepare datasets with optional augmentation.

    Parameters
    ----------
    data_config : DataConfig
        Data configuration.
    augmentation_config : AugmentationConfig
        Augmentation configuration.

    Returns
    -------
    tuple
        Four elements: (train_dataset, val_dataset, test_dataset, label_encoder)
        where label_encoder is fitted on training data.
    """
    logger.info("Loading datasets...")
    train_df, val_df, test_df = load_datasets(
        data_config.train_path,
        data_config.val_path,
        data_config.test_path
    )

    # Convert to Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Optional augmentation
    if augmentation_config.enabled:
        logger.info("Augmenting datasets...")
        train_dataset = augment_dataset(
            train_dataset,
            augmentation_config.augmentation_numbers,
            augmentation_config.tastes,
            data_config.label_column,
            data_config.smiles_column
        )
        val_dataset = augment_dataset(
            val_dataset,
            augmentation_config.augmentation_numbers,
            augmentation_config.tastes,
            data_config.label_column,
            data_config.smiles_column
        )
        test_dataset = augment_dataset(
            test_dataset,
            augmentation_config.augmentation_numbers,
            augmentation_config.tastes,
            data_config.label_column,
            data_config.smiles_column
        )

    return train_dataset, val_dataset, test_dataset


def save_run_summary(trainer, training_config: TrainingConfig) -> dict:
    """
    Save run summary to JSON file in the results folder organized by run_name.
    This captures the same metrics that wandb displays in the run summary.
    
    Parameters
    ----------
    trainer : Trainer
        Trained trainer instance.
    training_config : TrainingConfig
        Training configuration.
    
    Returns
    -------
    dict
        Dictionary containing the summary path and saved metrics.
    """
    # Determine output directory (same as trainer's output_dir)
    output_dir = trainer.args.output_dir
    
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract final metrics from trainer state
    state = trainer.state
    
    # Build summary dictionary with all metrics
    summary = {
        "run_name": training_config.run_name,
        "output_dir": output_dir,
        "training_config": {
            "num_train_epochs": training_config.num_train_epochs,
            "per_device_train_batch_size": training_config.per_device_train_batch_size,
            "per_device_eval_batch_size": training_config.per_device_eval_batch_size,
            "weight_decay": training_config.weight_decay,
            "evaluation_strategy": training_config.evaluation_strategy,
            "save_strategy": training_config.save_strategy,
        }
    }
    
    # Collect all metrics from log_history, keeping the most recent value for each key
    all_metrics = {}
    if hasattr(state, 'log_history') and state.log_history:
        for entry in state.log_history:
            for key, value in entry.items():
                # Keep the most recent value (later entries overwrite earlier ones)
                all_metrics[key] = value
    
    # Organize metrics similar to wandb summary format
    # Extract eval metrics (keys like eval_accuracy, eval_loss, eval_runtime, etc.)
    eval_metrics = {}
    train_metrics = {}
    other_metrics = {}
    
    for key, value in all_metrics.items():
        if key.startswith('eval_'):
            # Convert eval_accuracy -> accuracy, eval_loss -> loss, etc.
            clean_key = key.replace('eval_', '')
            eval_metrics[clean_key] = value
        elif key.startswith('train_'):
            clean_key = key.replace('train_', '')
            train_metrics[clean_key] = value
        elif key in ['loss', 'learning_rate', 'grad_norm', 'epoch', 'step']:
            # Training metrics without prefix
            train_metrics[key] = value
        else:
            other_metrics[key] = value
    
    # Add state information
    if hasattr(state, 'epoch'):
        train_metrics['epoch'] = state.epoch
    if hasattr(state, 'global_step'):
        train_metrics['global_step'] = state.global_step
    if hasattr(state, 'total_flos'):
        summary['total_flos'] = state.total_flos
    if hasattr(state, 'best_metric'):
        summary['best_metric'] = state.best_metric
    if hasattr(state, 'best_model_checkpoint'):
        summary['best_model_checkpoint'] = state.best_model_checkpoint
    
    # Organize final summary in wandb-like format
    summary['eval'] = eval_metrics
    summary['train'] = train_metrics
    if other_metrics:
        summary['other'] = other_metrics
    
    # Save to JSON file
    summary_path = os.path.join(output_dir, "run_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return {
        "summary_path": summary_path,
        "summary": summary
    }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FART Model Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for this training run (overrides default from config)"
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (overrides default from config)"
    )
    return parser.parse_args()


def main():
    """Main training and evaluation pipeline."""
    try:
        # Parse command-line arguments
        args = parse_args()
        
        # Initialize configurations with defaults
        model_config = ModelConfig()
        data_config = DataConfig()
        augmentation_config = AugmentationConfig()
        training_config = TrainingConfig()
        
        # Override with command-line arguments if provided
        if args.run_name is not None:
            training_config.run_name = args.run_name
        if args.model_checkpoint is not None:
            model_config.model_checkpoint = args.model_checkpoint

        logger.info("=" * 80)
        logger.info("FART Model Training Pipeline")
        logger.info("=" * 80)
        logger.info(f"Model checkpoint: {model_config.model_checkpoint}")
        logger.info(f"Tokenizer checkpoint: {model_config.tokenizer_checkpoint}")
        logger.info(f"Run name: {training_config.run_name}")
        logger.info(f"Augmentation enabled: {augmentation_config.enabled}")

        # Prepare datasets (without tokenization and label encoding)
        train_dataset, val_dataset, test_dataset = prepare_datasets(
            data_config,
            augmentation_config
        )

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")

        # Create trainer (this handles tokenization and label encoding in correct order)
        logger.info("Creating trainer...")
        trainer, test_dataset, label_encoder = create_trainer(
            model_config,
            training_config,
            train_dataset,
            val_dataset,
            test_dataset,
            data_config.label_column
        )

        logger.info(f"Number of classes: {len(label_encoder.classes_)}")
        logger.info(f"Class names: {label_encoder.classes_}")

        # Train model
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed!")

        # Save run summary to JSON file
        run_summary = save_run_summary(trainer, training_config)
        logger.info(f"Run summary saved to: {run_summary['summary_path']}")

        # Evaluate on validation set
        logger.info("Evaluating on validation set...")
        val_results = trainer.evaluate(eval_dataset=val_dataset)
        logger.info(f"Validation results: {val_results}")

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = evaluate_model(trainer, test_dataset, label_encoder)
        
        logger.info("=" * 80)
        logger.info("Test Set Evaluation Results")
        logger.info("=" * 80)
        print_evaluation_results(test_results, label_encoder)

    except Exception as e:
        logger.error(f"Error during training/evaluation: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
