import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
import random
import numpy as np


class WikiTextDataset(Dataset):
    """WikiText-2 dataset for BERT pre-training with MLM and NSP."""
    
    def __init__(self, split='train', max_length=128, mlm_probability=0.15):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        
        # Load WikiText-2 dataset
        print(f"Loading WikiText-2 {split} split...")
        dataset = load_dataset('wikitext', 'wikitext-2-v1', split=split)
        
        # Process into sentences (filter empty lines)
        self.sentences = []
        for text in dataset['text']:
            text = text.strip()
            if text and len(text) > 10:  # Filter very short lines
                self.sentences.append(text)
        
        print(f"Loaded {len(self.sentences)} sentences from {split} split")
        
        # Special token IDs
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.vocab_size = len(self.tokenizer)
    
    def __len__(self):
        return len(self.sentences)
    
    def create_sentence_pair(self, idx):
        """Create sentence pair with 50% next sentence and 50% random."""
        sentence_a = self.sentences[idx]
        
        # 50% of the time, use next sentence (is_next = 1)
        if random.random() < 0.5 and idx + 1 < len(self.sentences):
            sentence_b = self.sentences[idx + 1]
            is_next = 1
        else:
            # Random sentence (is_next = 0)
            random_idx = random.randint(0, len(self.sentences) - 1)
            while random_idx == idx or random_idx == idx + 1:
                random_idx = random.randint(0, len(self.sentences) - 1)
            sentence_b = self.sentences[random_idx]
            is_next = 0
        
        return sentence_a, sentence_b, is_next
    
    def tokenize_pair(self, sentence_a, sentence_b):
        """Tokenize sentence pair with [CLS] and [SEP] tokens."""
        # Tokenize both sentences
        tokens_a = self.tokenizer.tokenize(sentence_a)
        tokens_b = self.tokenizer.tokenize(sentence_b)
        
        # Truncate to fit max_length accounting for [CLS], [SEP], [SEP]
        max_tokens = self.max_length - 3
        while len(tokens_a) + len(tokens_b) > max_tokens:
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        
        # Build input: [CLS] + sentence_a + [SEP] + sentence_b + [SEP]
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        
        # Create segment IDs: 0 for sentence A, 1 for sentence B
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        
        # Convert to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        return input_ids, segment_ids, tokens
    
    def create_mlm_predictions(self, input_ids, tokens):
        """Create masked LM predictions following BERT masking strategy."""
        labels = [-100] * len(input_ids)  # -100 is ignored in loss
        
        # Don't mask [CLS], [SEP], or [PAD]
        special_tokens = {self.cls_token_id, self.sep_token_id, self.pad_token_id}
        
        # Get maskable positions
        maskable_positions = [
            i for i, token_id in enumerate(input_ids) 
            if token_id not in special_tokens
        ]
        
        # Select 15% of tokens to mask
        num_to_mask = max(1, int(len(maskable_positions) * self.mlm_probability))
        mask_positions = random.sample(maskable_positions, min(num_to_mask, len(maskable_positions)))
        
        for pos in mask_positions:
            labels[pos] = input_ids[pos]  # Store original token
            
            prob = random.random()
            if prob < 0.8:
                # 80% of the time: replace with [MASK]
                input_ids[pos] = self.mask_token_id
            elif prob < 0.9:
                # 10% of the time: replace with random token
                input_ids[pos] = random.randint(0, self.vocab_size - 1)
            # 10% of the time: keep original token
        
        return input_ids, labels
    
    def __getitem__(self, idx):
        """Get a single training example."""
        # Create sentence pair
        sentence_a, sentence_b, is_next = self.create_sentence_pair(idx)
        
        # Tokenize
        input_ids, segment_ids, tokens = self.tokenize_pair(sentence_a, sentence_b)
        
        # Create MLM predictions
        input_ids, mlm_labels = self.create_mlm_predictions(input_ids.copy(), tokens)
        
        # Pad to max_length
        padding_length = self.max_length - len(input_ids)
        input_ids += [self.pad_token_id] * padding_length
        segment_ids += [0] * padding_length
        mlm_labels += [-100] * padding_length
        attention_mask = [1] * (self.max_length - padding_length) + [0] * padding_length
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'token_type_ids': torch.tensor(segment_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'mlm_labels': torch.tensor(mlm_labels, dtype=torch.long),
            'nsp_label': torch.tensor(is_next, dtype=torch.long)
        }


def get_dataloaders(batch_size=16, max_length=128, mlm_probability=0.15):
    """Create train and validation dataloaders."""
    
    train_dataset = WikiTextDataset(
        split='train',
        max_length=max_length,
        mlm_probability=mlm_probability
    )
    
    val_dataset = WikiTextDataset(
        split='validation',
        max_length=max_length,
        mlm_probability=mlm_probability
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.tokenizer