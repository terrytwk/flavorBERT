""" Script for training a Roberta Masked-Language Model (Self-Contained)

Dependencies:
    pip install torch transformers tokenizers

Usage [SMILES tokenizer]:
    python train_roberta_mlm_simple.py --dataset_path="data.txt" --output_dir="output" --tokenizer_type=smiles

Usage [BPE tokenizer (trains new tokenizer)]:
    python train_roberta_mlm_simple.py --dataset_path="data.txt" --output_dir="output" --tokenizer_type=bpe
"""
import os
import argparse
import torch
from torch.utils.data import Dataset, random_split

from transformers import (
    RobertaConfig,
    RobertaTokenizerFast,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from tokenizers import ByteLevelBPETokenizer

# --- Custom Dataset Class (Replaces external dependency) ---
class LineByLineTextDataset(Dataset):
    """
    Reads a file line-by-line and tokenizes it on the fly.
    This is memory efficient and requires no external libraries.
    """
    def __init__(self, tokenizer, file_path, block_size=512):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        print(f"Reading lines from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read all lines into memory (fast enough for <10M lines on modern servers)
            # Filter empty lines
            self.lines = [line.strip() for line in f if line.strip()]
            
        print(f"Loaded {len(self.lines)} valid lines.")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        # Tokenize the line
        # We return a list of IDs; the DataCollator will pad them into a batch later
        return self.tokenizer(
            line, 
            truncation=True, 
            max_length=self.block_size,
            add_special_tokens=True
        )["input_ids"]


def main():
    parser = argparse.ArgumentParser(description="Train RoBERTa MLM from scratch")
    
    # Required parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to text file (one molecule/sentence per line)")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save the model")
    
    # Run configuration
    parser.add_argument("--run_name", type=str, default="roberta_mlm", help="Name for the training run")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the output directory")
    
    # Model configuration
    parser.add_argument("--vocab_size", type=int, default=600, help="Vocabulary size (ignored if using pre-trained tokenizer)")
    parser.add_argument("--max_position_embeddings", type=int, default=514, help="Max sequence length + 2")
    parser.add_argument("--num_attention_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--num_hidden_layers", type=int, default=6, help="Number of hidden layers")
    parser.add_argument("--type_vocab_size", type=int, default=1, help="Type vocab size")
    
    # Tokenizer configuration
    parser.add_argument("--tokenizer_type", type=str, default="smiles", choices=["smiles", "bpe"], help="Type of tokenizer")
    parser.add_argument("--tokenizer_path", type=str, default="seyonec/SMILES_tokenized_PubChem_shard00_160k", help="HuggingFace path for existing tokenizer")
    parser.add_argument("--BPE_min_frequency", type=int, default=2, help="Min frequency for BPE")
    parser.add_argument("--output_tokenizer_dir", type=str, default="./tokenizer_dir", help="Where to save trained BPE tokenizer")
    parser.add_argument("--max_tokenizer_len", type=int, default=512, help="Max length for tokenization")
    
    # Training configuration
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask")
    parser.add_argument("--frac_train", type=float, default=0.95, help="Fraction of data for training")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X steps")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Evaluate every X steps")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU execution")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(42)
    
    # Check device
    if args.no_cuda:
        device = torch.device("cpu")
        n_gpu = 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    
    print(f"Device: {device}, GPUs available: {n_gpu}")

    # 1. Setup Tokenizer
    if args.tokenizer_type == "smiles":
        print(f"Loading pre-trained tokenizer: {args.tokenizer_path}")
        tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer_path, max_len=args.max_tokenizer_len)
    else:
        # BPE Training logic
        tokenizer_path = args.output_tokenizer_dir
        if not os.path.exists(tokenizer_path):
            os.makedirs(tokenizer_path)
        
        print(f"Training BPE tokenizer on {args.dataset_path}...")
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=args.dataset_path, 
            vocab_size=args.vocab_size, 
            min_frequency=args.BPE_min_frequency, 
            special_tokens=["<s>","<pad>","</s>","<unk>","<mask>"]
        )
        tokenizer.save_model(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")
        # Reload as RobertaTokenizerFast
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_len=args.max_tokenizer_len)

    # 2. Configure Model
    print("Initializing Model Config...")
    
    # Safety check for vocab size
    real_vocab_size = max(tokenizer.vocab_size, len(tokenizer))
    # Ensure config vocab size is large enough
    config_vocab_size = max(args.vocab_size, real_vocab_size + 128)

    config = RobertaConfig(
        vocab_size=config_vocab_size,
        max_position_embeddings=args.max_position_embeddings,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        type_vocab_size=args.type_vocab_size,
    )

    model = RobertaForMaskedLM(config=config)
    print(f"Model initialized with {model.num_parameters():,} parameters.")

    # 3. Load Dataset
    print("Preparing Dataset...")
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer, 
        file_path=args.dataset_path, 
        block_size=args.max_tokenizer_len
    )

    # Split Train/Eval
    train_size = int(args.frac_train * len(dataset))
    eval_size = len(dataset) - train_size
    print(f"Train samples: {train_size}, Eval samples: {eval_size}")
    
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    # Data Collator (Handles masking and padding)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability
    )

    # 4. Training Setup
    output_run_dir = os.path.join(args.output_dir, args.run_name)
    
    training_args = TrainingArguments(
        output_dir=output_run_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        
        # Evaluation & Saving
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        
        # Logging
        logging_steps=args.logging_steps,
        report_to="none",  # Disable external loggers
        
        # Optimization
        fp16=args.fp16 and torch.cuda.is_available(),
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # 5. Train
    print("\n" + "="*40)
    print("STARTING TRAINING")
    print("="*40)
    trainer.train()
    
    # 6. Save
    print("\n" + "="*40)
    print(f"SAVING MODEL TO: {output_run_dir}/final")
    print("="*40)
    final_path = os.path.join(output_run_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print("Done!")

if __name__ == "__main__":
    main()