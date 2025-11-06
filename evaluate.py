# evaluate.py
import torch
from model import MiniBERT
from data import get_dataloaders
from transformers import BertTokenizer
from utils import get_device, predict_masked_tokens, evaluate_nsp, create_mlm_examples, create_nsp_examples
import torch.nn as nn
from tqdm import tqdm

device = get_device()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load model (adjust hidden_size to match your checkpoint!)
model = MiniBERT(
    vocab_size=len(tokenizer),
    hidden_size=512,  # CHANGE THIS to match your model
    num_layers=4,
    num_attention_heads=8,
    max_position_embeddings=128,
    intermediate_size=1024
).to(device)

checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded model from epoch {checkpoint['epoch']}")

# Get validation data
_, val_loader, _ = get_dataloaders(batch_size=16, max_length=128)

# Evaluate on validation set
print("\n=== Validation Metrics ===")
total_loss = 0
total_mlm_loss = 0
total_nsp_loss = 0
total_nsp_correct = 0
total_nsp_samples = 0
total_mlm_correct = 0
total_mlm_tokens = 0

mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
nsp_criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    for batch in tqdm(val_loader):
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        mlm_labels = batch['mlm_labels'].to(device)
        nsp_labels = batch['nsp_label'].to(device)
        
        mlm_logits, nsp_logits = model(input_ids, token_type_ids, attention_mask)
        
        # Losses
        mlm_loss = mlm_criterion(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1))
        nsp_loss = nsp_criterion(nsp_logits, nsp_labels)
        
        total_mlm_loss += mlm_loss.item()
        total_nsp_loss += nsp_loss.item()
        total_loss += (mlm_loss + nsp_loss).item()
        
        # NSP accuracy
        nsp_preds = torch.argmax(nsp_logits, dim=1)
        total_nsp_correct += (nsp_preds == nsp_labels).sum().item()
        total_nsp_samples += nsp_labels.size(0)
        
        # MLM accuracy
        mask = mlm_labels != -100
        if mask.sum() > 0:
            preds = torch.argmax(mlm_logits, dim=-1)
            correct = (preds == mlm_labels) & mask
            total_mlm_correct += correct.sum().item()
            total_mlm_tokens += mask.sum().item()

# Print metrics
print(f"\nTotal Loss: {total_loss/len(val_loader):.4f}")
print(f"MLM Loss: {total_mlm_loss/len(val_loader):.4f}")
print(f"NSP Loss: {total_nsp_loss/len(val_loader):.4f}")
print(f"NSP Accuracy: {100.0*total_nsp_correct/total_nsp_samples:.2f}%")
print(f"MLM Accuracy: {100.0*total_mlm_correct/total_mlm_tokens:.2f}%")
print(f"Perplexity: {torch.exp(torch.tensor(total_mlm_loss/len(val_loader))):.2f}")

# Example predictions
print("\n=== MLM Examples ===")
for ex in create_mlm_examples(tokenizer):
    print(f"\nInput: {ex}")
    preds = predict_masked_tokens(model, tokenizer, ex, device, top_k=3)
    for p in preds:
        for token, prob in p['predictions']:
            print(f"  → {token} ({prob:.3f})")

print("\n=== NSP Examples ===")
for ex in create_nsp_examples():
    pred, conf, _ = evaluate_nsp(model, ex['sentence_a'], ex['sentence_b'], tokenizer, device)
    result = "✓" if pred == ex['label'] else "✗"
    print(f"{result} {ex['sentence_a'][:40]}... | {'Next' if pred==1 else 'NotNext'} ({conf:.2f})")