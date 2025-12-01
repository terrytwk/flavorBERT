"""
Multi-task RoBERTa model: MLM + Food Co-occurrence Prediction
"""

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig
from typing import Optional


class RobertaMLMAndFoodHead(nn.Module):
    """
    RoBERTa with two heads:
    1. MLM head (standard masked language modeling)
    2. Food prediction head (multi-label classification)
    """
    
    def __init__(self, config: RobertaConfig, food_vocab_size: int):
        super().__init__()
        self.config = config
        self.food_vocab_size = food_vocab_size
        
        # Shared RoBERTa encoder
        self.roberta = RobertaModel(config)
        
        # MLM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.roberta.embeddings.word_embeddings.weight
        
        # Food prediction head
        self.food_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, food_vocab_size)
        )
        
        self.mlm_loss_fct = nn.CrossEntropyLoss()
        self.food_loss_fct = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        food_labels: Optional[torch.Tensor] = None,
        has_food_context: Optional[torch.Tensor] = None,
        mlm_weight: float = 1.0,
        food_weight: float = 1.0,
    ):
        # Encode
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]  # [CLS] token
        
        # MLM predictions
        mlm_logits = self.lm_head(sequence_output)
        
        # Food predictions
        food_logits = self.food_head(pooled_output)
        
        # Calculate losses
        total_loss = None
        mlm_loss = None
        food_loss = None
        
        if labels is not None:
            mlm_loss = self.mlm_loss_fct(
                mlm_logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )
            total_loss = mlm_weight * mlm_loss
        
        if food_labels is not None and has_food_context is not None:
            if has_food_context.any():
                food_logits_masked = food_logits[has_food_context]
                food_labels_masked = food_labels[has_food_context]
                food_loss = self.food_loss_fct(food_logits_masked, food_labels_masked)
                
                if total_loss is not None:
                    total_loss = total_loss + food_weight * food_loss
                else:
                    total_loss = food_weight * food_loss
        
        return {
            'loss': total_loss,
            'mlm_loss': mlm_loss,
            'food_loss': food_loss,
            'mlm_logits': mlm_logits,
            'food_logits': food_logits,
        }
    
    def save_pretrained(self, save_directory):
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        with open(os.path.join(save_directory, "food_config.json"), 'w') as f:
            json.dump({"food_vocab_size": self.food_vocab_size}, f)
    
    @classmethod
    def from_pretrained(cls, pretrained_path, config=None):
        import os
        import json
        
        if config is None:
            config = RobertaConfig.from_pretrained(pretrained_path)
        
        with open(os.path.join(pretrained_path, "food_config.json"), 'r') as f:
            food_config = json.load(f)
        
        model = cls(config, food_config['food_vocab_size'])
        state_dict = torch.load(os.path.join(pretrained_path, "pytorch_model.bin"))
        model.load_state_dict(state_dict)
        
        return model

