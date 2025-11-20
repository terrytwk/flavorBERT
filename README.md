# FlavorBERT: A Foundational Model for Flavor Prediction

FlavorBERT is a foundational transformer model for predicting flavor properties from molecular structures. This project builds upon [ChemBERTa](https://github.com/seyonechithrananda/bert-loves-chemistry), a BERT-like model for chemical SMILES, and applies it to flavor prediction tasks using the [FART (Flavor and Aroma Recognition Task)](https://github.com/fart-lab/fart) dataset framework. The model combines both taste and smell prediction capabilities, providing a comprehensive approach to flavor analysis.

## Overview

Flavor is the complex perception that combines taste (sweet, bitter, sour, umami) and smell (aroma), and is crucial for food science, flavor compound discovery, and rational food design. FlavorBERT addresses this challenge by:

1. **Pre-training** a RoBERTa-based transformer model on large-scale molecular datasets (FoodDB, PubChem) using masked language modeling (MLM)
2. **Fine-tuning** the pre-trained model on various molecular property prediction tasks
3. **Evaluation** on flavor-specific datasets (FART) to predict taste categories from chemical structures

The architecture leverages the attention mechanism to learn molecular representations that capture structural features relevant to flavor perception.

## Related Work

This project builds upon:

- **ChemBERTa**: A collection of BERT-like models applied to chemical SMILES data for drug design and chemical property prediction
  - Repository: [bert-loves-chemistry](https://github.com/seyonechithrananda/bert-loves-chemistry)
  - Paper: [ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction](https://arxiv.org/abs/2010.09885)

- **FART**: Flavor and Aroma Recognition Task dataset and models for molecular taste prediction
  - Repository: [fart-lab/fart](https://github.com/fart-lab/fart/tree/main)
  - Paper: [FART: A Transformer-Based Model for Flavor Prediction](https://www.nature.com/articles/s41538-025-00474-z) (Nature Scientific Reports)

## Installation

### Prerequisites

- Conda (for environment management)
- CUDA-capable GPU (recommended for training)

### Setup

1. **Create the conda environment:**

   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment:**

   ```bash
   conda activate flavorbert_env
   ```

3. **Install the package in development mode:**

   ```bash
   pip install -e .
   ```

4. **Set up Weights & Biases (optional, for experiment tracking):**

   ```bash
   wandb login
   ```

## Usage

### 1. Pre-training ChemBERTa on Molecular Data

Train a RoBERTa model using masked language modeling on molecular SMILES data. **Important**: Change `run_name` and `output_dir` for each new training run.

```bash
python chemberta/train/train_roberta.py \
    --model_type=mlm \
    --dataset_path=chemberta/data/foodb_1k_smiles.txt \
    --output_dir=chemberta/train/my_training4 \
    --run_name=test_run \
    --per_device_train_batch_size=4 \
    --num_hidden_layers=6 \
    --num_attention_heads=12 \
    --num_train_epochs=10 \
    --eval_steps=50 \
    --save_steps=50
```

**Parameters:**

- `--model_type`: Model type (use `mlm` for masked language modeling)
- `--dataset_path`: Path to SMILES dataset file (one SMILES string per line)
- `--output_dir`: Directory to save model checkpoints
- `--run_name`: Experiment name for logging
- `--per_device_train_batch_size`: Batch size per device
- `--num_hidden_layers`: Number of transformer layers
- `--num_attention_heads`: Number of attention heads
- `--num_train_epochs`: Number of training epochs
- `--eval_steps`: Steps between evaluations
- `--save_steps`: Steps between model checkpoints

### 2. Fine-tuning on Molecular Property Prediction Tasks

Fine-tune the pre-trained model on multiple molecular property prediction datasets from DeepChem's MoleculeNet. **Important**: Match `output_dir` and `run_name` with your pre-training configuration.

```bash
python chemberta/finetune/finetune.py \
    --datasets=bace_classification,bace_regression,bbbp,clearance,clintox,delaney,lipo,tox21 \
    --output_dir=chemberta/finetune/my_dir \
    --run_name=my_run \
    --pretrained_model_name_or_path="chemberta/train/my_training4/test_run/final" \
    --n_trials=20 \
    --n_seeds=5
```

**Parameters:**

- `--datasets`: Comma-separated list of MoleculeNet datasets
- `--output_dir`: Directory to save fine-tuning results
- `--run_name`: Experiment name
- `--pretrained_model_name_or_path`: Path to pre-trained model checkpoint
- `--n_trials`: Number of hyperparameter optimization trials
- `--n_seeds`: Number of random seeds for evaluation

**Available Datasets:**

- `bace_classification`: Binding affinity classification
- `bace_regression`: Binding affinity regression
- `bbbp`: Blood-brain barrier penetration
- `clearance`: Molecular clearance prediction
- `clintox`: Clinical toxicity
- `delaney`: Aqueous solubility
- `lipo`: Lipophilicity
- `tox21`: Toxicology challenge

### 3. FART Evaluation (Flavor Prediction)

Evaluate the model on the FART dataset for flavor/taste prediction tasks:

```bash
python fart/models/FART_Models.py \
    --run_name my_experiment \
    --model_checkpoint chemberta/train/my_training4/test_run/final
```

**Parameters:**

- `--run_name`: Experiment name for logging
- `--model_checkpoint`: Path to pre-trained or fine-tuned model checkpoint

The FART evaluation predicts taste categories (sweet, bitter, sour, umami, undefined) from molecular SMILES strings.

## Project Structure

```text
flavorBERT/
├── chemberta/              # ChemBERTa model implementation
│   ├── train/              # Pre-training scripts
│   ├── finetune/           # Fine-tuning scripts
│   ├── data/               # Training datasets
│   ├── utils/              # Utility functions
│   └── bertviz_clone/      # Attention visualization tools
├── fart/                   # FART evaluation framework
│   ├── models/             # Model training and evaluation
│   ├── dataset/            # FART dataset and splits
│   └── plots/              # Visualization scripts
├── environment.yml         # Conda environment specification
└── setup.py               # Package setup configuration
```

## Datasets

### Pre-training Datasets

- **FoodDB SMILES**: Food-related molecular structures
- **PubChem**: Large-scale chemical database subsets
- Located in: `chemberta/data/`

### Fine-tuning Datasets

- **MoleculeNet**: Standard benchmark datasets for molecular property prediction
- Automatically downloaded when running fine-tuning scripts

### Flavor Datasets

- **FART Dataset**: 15,025 curated molecular tastants with taste labels (sweet, bitter, sour, umami, undefined)
- Located in: `fart/dataset/`
- For more details, see the [FART repository](https://github.com/fart-lab/fart)

## Model Architecture

FlavorBERT uses a RoBERTa-based transformer architecture:

- **Base Model**: RoBERTa (Robustly Optimized BERT Pretraining Approach)
- **Input**: SMILES strings (text representation of molecular structures)
- **Pre-training Task**: Masked Language Modeling (MLM)
- **Fine-tuning Tasks**: Classification and regression for molecular properties
- **Flavor Prediction**: Multi-label classification for taste categories

## Citation

If you use FlavorBERT in your research, please cite:

```bibtex
@article{chemberta2020,
  title={ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction},
  author={Chithrananda, Seyone and Grand, Gabriel and Ramsundar, Bharath},
  journal={arXiv preprint arXiv:2010.09885},
  year={2020}
}

@article{fart2025,
  title={FART: A Transformer-Based Model for Flavor Prediction},
  author={...},
  journal={Nature Scientific Reports},
  year={2025},
  doi={https://www.nature.com/articles/s41538-025-00474-z}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- [ChemBERTa](https://github.com/seyonechithrananda/bert-loves-chemistry) for the foundational model architecture
- [FART Lab](https://github.com/fart-lab/fart) for the flavor dataset and evaluation framework
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for the transformer implementation
- [DeepChem](https://github.com/deepchem/deepchem) for molecular datasets and tools

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.
