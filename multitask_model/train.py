"""
Training script for multi-task RoBERTa model
"""

import os
import argparse
import torch
from transformers import RobertaConfig, RobertaTokenizerFast, TrainingArguments, Trainer

from model import RobertaMLMAndFoodHead
from dataset import MultiTaskChemDataset, MultiTaskCollator


class MultiTaskTrainer(Trainer):
    """Custom trainer for multi-task loss."""
    
    def __init__(self, mlm_weight=1.0, food_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlm_weight = mlm_weight
        self.food_weight = food_weight
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=inputs['labels'],
            food_labels=inputs['food_labels'],
            has_food_context=inputs['has_food_context'],
            mlm_weight=self.mlm_weight,
            food_weight=self.food_weight,
        )
        
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--general_chem_data', type=str, required=True)
    parser.add_argument('--food_context_data', type=str, required=True)
    parser.add_argument('--food_vocab', type=str, required=True)
    
    # Model
    parser.add_argument('--tokenizer_path', type=str, 
                       default='seyonec/SMILES_tokenized_PubChem_shard00_160k')
    parser.add_argument('--num_hidden_layers', type=int, default=6)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--hidden_size', type=int, default=768)
    
    # Training
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--run_name', type=str, default='multitask')
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--per_device_train_batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--save_steps', type=int, default=1500)
    parser.add_argument('--logging_steps', type=int, default=100)
    
    # Multi-task
    parser.add_argument('--mlm_probability', type=float, default=0.15)
    parser.add_argument('--food_task_weight', type=float, default=0.3)
    parser.add_argument('--mlm_loss_weight', type=float, default=1.0)
    parser.add_argument('--food_loss_weight', type=float, default=1.0)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Multi-Task RoBERTa Training")
    print("=" * 80)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {args.tokenizer_path}")
    tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer_path, max_len=512)
    
    # Load dataset
    print("\nLoading datasets...")
    dataset = MultiTaskChemDataset(
        general_smiles_file=args.general_chem_data,
        food_context_file=args.food_context_data,
        food_vocab_file=args.food_vocab,
        food_task_weight=args.food_task_weight,
    )
    
    # Load food vocab size
    import json
    with open(args.food_vocab, 'r') as f:
        food_vocab = json.load(f)
        food_vocab_size = food_vocab['vocab_size']
    
    # Create collator
    data_collator = MultiTaskCollator(tokenizer, mlm_probability=args.mlm_probability)
    
    # Create model
    print("\nCreating model...")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Tokenizer len: {len(tokenizer)}")
    
    # Ensure vocab size is large enough to hold all tokens
    # Add a small buffer for safety in case of off-by-one errors or special tokens
    safe_vocab_size = max(tokenizer.vocab_size, len(tokenizer)) + 10
    print(f"Using safe vocab size: {safe_vocab_size}")
    
    config = RobertaConfig(
        vocab_size=safe_vocab_size,
        max_position_embeddings=514,  # RoBERTa usually needs +2 for special tokens
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        hidden_size=args.hidden_size,
        intermediate_size=4 * args.hidden_size,
        type_vocab_size=1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    model = RobertaMLMAndFoodHead(config, food_vocab_size)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=3,
        overwrite_output_dir=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        remove_unused_columns=False,
        save_safetensors=False,  # Disable safetensors to avoid shared weight errors
    )
    
    # Create trainer
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        mlm_weight=args.mlm_loss_weight,
        food_weight=args.food_loss_weight,
    )
    
    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    trainer.train()
    
    # Save
    final_path = os.path.join(args.output_dir, "final")
    print(f"\nSaving model to {final_path}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()

