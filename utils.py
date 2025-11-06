import torch
import random
import numpy as np
import os


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def predict_masked_tokens(model, tokenizer, text, device, top_k=5):
    """
    Predict masked tokens in text.
    
    Args:
        model: Trained BERT model
        tokenizer: Tokenizer
        text: Input text with [MASK] tokens
        device: Device to run on
        top_k: Number of top predictions to return
    
    Returns:
        List of (position, original_token, predictions) tuples
    """
    model.eval()
    
    # Tokenize
    tokens = tokenizer.tokenize(text)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    
    # Find mask positions
    mask_positions = [i for i, token in enumerate(tokens) if token == '[MASK]']
    
    if not mask_positions:
        return []
    
    # Convert to IDs
    input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)]).to(device)
    
    with torch.no_grad():
        mlm_logits, _ = model(input_ids)
    
    predictions = []
    for pos in mask_positions:
        # Get top k predictions
        logits = mlm_logits[0, pos]
        top_k_indices = torch.topk(logits, top_k).indices.cpu().numpy()
        top_k_tokens = [tokenizer.convert_ids_to_tokens([idx])[0] for idx in top_k_indices]
        top_k_probs = torch.softmax(logits, dim=-1)[top_k_indices].cpu().numpy()
        
        predictions.append({
            'position': pos,
            'predictions': list(zip(top_k_tokens, top_k_probs))
        })
    
    return predictions

def calculate_perplexity(loss):
    """Calculate perplexity from loss."""
    return torch.exp(torch.tensor(loss)).item()

def evaluate_nsp(model, sentence_a, sentence_b, tokenizer, device):
    """
    Evaluate next sentence prediction.
    
    Args:
        model: Trained BERT model
        sentence_a: First sentence
        sentence_b: Second sentence
        tokenizer: Tokenizer
        device: Device to run on
    
    Returns:
        Prediction (0 = not next, 1 = is next) and confidence
    """
    model.eval()
    
    # Tokenize both sentences
    tokens_a = tokenizer.tokenize(sentence_a)
    tokens_b = tokenizer.tokenize(sentence_b)
    
    # Build input
    tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Create segment IDs
    segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
    
    # Convert to tensors
    input_ids = torch.tensor([input_ids]).to(device)
    segment_ids = torch.tensor([segment_ids]).to(device)
    
    with torch.no_grad():
        _, nsp_logits = model(input_ids, token_type_ids=segment_ids)
    
    probs = torch.softmax(nsp_logits, dim=1)[0]
    prediction = torch.argmax(probs).item()
    confidence = probs[prediction].item()
    
    return prediction, confidence, probs.cpu().numpy()


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds):
    """Format seconds into readable time string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_device():
    """Get available device (cuda/mps/cpu)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def create_mlm_examples(tokenizer, num_examples=5):
    """Create example sentences for MLM evaluation."""
    examples = [
        "The capital of France [MASK] Paris.",
        "The quick brown fox jumped over the lazy [MASK].",
        "Albert Einstein was a famous [MASK].",
        "The sun rises [MASK] the East.",
        "Water [MASK] at 100 degrees Celsius."
    ]
    return examples[:num_examples]


def create_nsp_examples():
    """Create example sentence pairs for NSP evaluation."""
    examples = [
        {
            'sentence_a': "The weather is beautiful today.",
            'sentence_b': "I think I'll go for a walk.",
            'label': 1  # Next sentence
        },
        {
            'sentence_a': "The weather is beautiful today.",
            'sentence_b': "The movie was not interesting.",
            'label': 0  # Not next sentence
        },
        {
            'sentence_a': "She opened the book.",
            'sentence_b': "The first chapter was about history.",
            'label': 1
        },
        {
            'sentence_a': "She opened the book.",
            'sentence_b': "Pizza is my favorite food.",
            'label': 0
        },
        {
            'sentence_a': "The team won the championship.",
            'sentence_b': "They celebrated all night long.",
            'label': 1
        }
    ]
    return examples