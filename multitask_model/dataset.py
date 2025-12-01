"""
Dataset for multi-task training: General chemistry (MLM) + FoodDB (MLM + food prediction)
"""

import json
import random
import torch
from torch.utils.data import Dataset


class MultiTaskChemDataset(Dataset):
    """
    Combines general chemistry SMILES and FoodDB SMILES with food context.
    
    For each sample returns:
    - smiles: SMILES string
    - food_labels: Multi-hot vector of foods (or None)
    - has_food_context: Boolean
    """
    
    def __init__(
        self,
        general_smiles_file: str,
        food_context_file: str,
        food_vocab_file: str,
        food_task_weight: float = 0.3,
    ):
        """
        Args:
            general_smiles_file: Path to general chemistry SMILES (one per line)
            food_context_file: Path to FoodDB context JSONL
            food_vocab_file: Path to food vocabulary JSON
            food_task_weight: Probability of sampling FoodDB vs general chemistry
        """
        self.food_task_weight = food_task_weight
        
        # Load general chemistry SMILES
        print(f"Loading general chemistry SMILES from {general_smiles_file}...")
        with open(general_smiles_file, 'r') as f:
            self.general_smiles = [line.strip() for line in f if line.strip()]
        print(f"  Loaded {len(self.general_smiles)} general chemistry SMILES")
        
        # Load FoodDB context data
        print(f"Loading FoodDB context from {food_context_file}...")
        with open(food_context_file, 'r') as f:
            self.food_context_data = [json.loads(line) for line in f]
        print(f"  Loaded {len(self.food_context_data)} FoodDB compounds with context")
        
        # Load food vocabulary
        with open(food_vocab_file, 'r') as f:
            vocab_data = json.load(f)
            self.food_vocab_size = vocab_data['vocab_size']
        
        print(f"  Food vocabulary size: {self.food_vocab_size}")
    
    def __len__(self):
        return len(self.general_smiles) + len(self.food_context_data)
    
    def __getitem__(self, idx):
        """Return SMILES and food labels (if applicable)."""
        # Randomly choose between FoodDB and general chemistry
        use_food_context = random.random() < self.food_task_weight
        
        if use_food_context and len(self.food_context_data) > 0:
            # Sample from FoodDB
            record = random.choice(self.food_context_data)
            smiles = record['smiles']
            food_indices = record['food_indices']
            
            # Create multi-hot food label vector
            food_labels = torch.zeros(self.food_vocab_size, dtype=torch.float32)
            for food_idx in food_indices:
                food_labels[food_idx] = 1.0
            
            has_food_context = True
        else:
            # Sample from general chemistry
            smiles = random.choice(self.general_smiles)
            food_labels = torch.zeros(self.food_vocab_size, dtype=torch.float32)
            has_food_context = False
        
        return {
            'smiles': smiles,
            'food_labels': food_labels,
            'has_food_context': has_food_context,
        }


class MultiTaskCollator:
    """
    Collates batches and applies MLM masking.
    """
    
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
    
    def __call__(self, examples):
        """Tokenize SMILES and apply MLM masking."""
        # Extract SMILES
        smiles_list = [ex['smiles'] for ex in examples]
        
        # Tokenize
        encoding = self.tokenizer(
            smiles_list,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        # Apply MLM masking
        input_ids, labels = self.mask_tokens(input_ids)
        
        # Stack food labels
        food_labels = torch.stack([ex['food_labels'] for ex in examples])
        has_food_context = torch.tensor([ex['has_food_context'] for ex in examples], dtype=torch.bool)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'food_labels': food_labels,
            'has_food_context': has_food_context,
        }
    
    def mask_tokens(self, inputs):
        """Apply MLM masking (80% mask, 10% random, 10% keep)."""
        labels = inputs.clone()
        
        # Create probability matrix
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # Don't mask special tokens
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens
        
        # 80% mask
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id
        
        # 10% random
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        return inputs, labels

