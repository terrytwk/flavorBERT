"""
Configuration settings for FART model training and evaluation.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model and tokenizer configuration."""
    model_checkpoint: str = "./chemberta/100k_test/final"
    tokenizer_checkpoint: str = "seyonec/SMILES_tokenized_PubChem_shard00_160k"
    num_labels: int = 5


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    train_path: str = "../dataset/splits/fart_train.csv"
    val_path: str = "../dataset/splits/fart_val.csv"
    test_path: str = "../dataset/splits/fart_test.csv"
    smiles_column: str = "Canonicalized SMILES"
    label_column: str = "Canonicalized Taste"
    max_length: int = 512


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    enabled: bool = False
    tastes: List[str] = None
    augmentation_numbers: List[int] = None

    def __post_init__(self):
        """Set default values if None."""
        if self.tastes is None:
            self.tastes = ['bitter', 'sour', 'sweet', 'umami', 'undefined']
        if self.augmentation_numbers is None:
            self.augmentation_numbers = [10, 10, 10, 10, 10]


@dataclass
class TrainingConfig:
    """Training arguments configuration."""
    output_dir: str = "./results"
    logging_dir: str = "./logs"
    run_name: Optional[str] = "<RUN_NAME>"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    weight_decay: float = 0.01
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    save_total_limit: int = 5

