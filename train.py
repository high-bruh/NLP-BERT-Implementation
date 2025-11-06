import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from utils import save_checkpoint, load_checkpoint


class BERTTrainer:
    """Trainer for Mini BERT with MLM and NSP objectives."""
    
    def __init__(self, model, train_loader, val_loader, tokenizer, device, lr=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Loss functions
        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.nsp_criterion = nn.CrossEntropyLoss()
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_nsp_accuracies = []
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_mlm_loss = 0
        total_nsp_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            mlm_labels = batch['mlm_labels'].to(self.device)
            nsp_labels = batch['nsp_label'].to(self.device)
            
            # Forward pass
            mlm_logits, nsp_logits = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )
            
            # Calculate losses
            mlm_loss = self.mlm_criterion(
                mlm_logits.view(-1, mlm_logits.size(-1)),
                mlm_labels.view(-1)
            )
            nsp_loss = self.nsp_criterion(nsp_logits, nsp_labels)
            
            # Combined loss
            loss = mlm_loss + nsp_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            total_mlm_loss += mlm_loss.item()
            total_nsp_loss += nsp_loss.item()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mlm': f'{mlm_loss.item():.4f}',
                    'nsp': f'{nsp_loss.item():.4f}'
                })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_mlm_loss = total_mlm_loss / len(self.train_loader)
        avg_nsp_loss = total_nsp_loss / len(self.train_loader)
        
        self.train_losses.append(avg_loss)
        
        return avg_loss, avg_mlm_loss, avg_nsp_loss
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_mlm_loss = 0
        total_nsp_loss = 0
        total_nsp_correct = 0
        total_nsp_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                mlm_labels = batch['mlm_labels'].to(self.device)
                nsp_labels = batch['nsp_label'].to(self.device)
                
                # Forward pass
                mlm_logits, nsp_logits = self.model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask
                )
                
                # Calculate losses
                mlm_loss = self.mlm_criterion(
                    mlm_logits.view(-1, mlm_logits.size(-1)),
                    mlm_labels.view(-1)
                )
                nsp_loss = self.nsp_criterion(nsp_logits, nsp_labels)
                loss = mlm_loss + nsp_loss
                
                # Track losses
                total_loss += loss.item()
                total_mlm_loss += mlm_loss.item()
                total_nsp_loss += nsp_loss.item()
                
                # Calculate NSP accuracy
                nsp_preds = torch.argmax(nsp_logits, dim=1)
                total_nsp_correct += (nsp_preds == nsp_labels).sum().item()
                total_nsp_samples += nsp_labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_mlm_loss = total_mlm_loss / len(self.val_loader)
        avg_nsp_loss = total_nsp_loss / len(self.val_loader)
        nsp_accuracy = 100.0 * total_nsp_correct / total_nsp_samples
        
        self.val_losses.append(avg_loss)
        self.val_nsp_accuracies.append(nsp_accuracy)
        
        return avg_loss, avg_mlm_loss, avg_nsp_loss, nsp_accuracy
    
    def train(self, num_epochs, save_path='checkpoints'):
        """Train the model for multiple epochs."""
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"{'='*60}\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_mlm, train_nsp = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f} | MLM: {train_mlm:.4f} | NSP: {train_nsp:.4f}")
            
            # Validate
            val_loss, val_mlm, val_nsp_loss, nsp_acc = self.validate()
            print(f"Val Loss: {val_loss:.4f} | MLM: {val_mlm:.4f} | NSP: {val_nsp_loss:.4f}")
            print(f"NSP Accuracy: {nsp_acc:.2f}%")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    f"{save_path}/best_model.pt"
                )
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            
            # Save checkpoint every epoch
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                val_loss,
                f"{save_path}/checkpoint_epoch_{epoch}.pt"
            )
        
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final NSP accuracy: {self.val_nsp_accuracies[-1]:.2f}%")
        print(f"{'='*60}\n")
        
        return self.train_losses, self.val_losses, self.val_nsp_accuracies