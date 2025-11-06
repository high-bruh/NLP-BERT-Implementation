import torch
import argparse
import time
from model import MiniBERT
from data import get_dataloaders
from train import BERTTrainer
from utils import (
    set_seed, get_device, count_parameters, format_time,
    predict_masked_tokens, evaluate_nsp, create_mlm_examples, create_nsp_examples
)


def train_model(args):
    """Train the Mini BERT model."""
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, tokenizer = get_dataloaders(
        batch_size=args.batch_size,
        max_length=args.max_length,
        mlm_probability=args.mlm_probability
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = MiniBERT(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_length,
        dropout=args.dropout
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Initialize trainer
    trainer = BERTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        lr=args.learning_rate
    )
    
    # Train
    start_time = time.time()
    train_losses, val_losses, nsp_accuracies = trainer.train(
        num_epochs=args.num_epochs,
        save_path=args.save_dir
    )
    training_time = time.time() - start_time
    
    print(f"\nTotal training time: {format_time(training_time)}")
    
    return model, tokenizer, device


def evaluate_model(model, tokenizer, device):
    """Evaluate the model with examples."""
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    # Masked Language Modeling Examples
    print("\n--- Masked Language Modeling Examples ---\n")
    mlm_examples = create_mlm_examples(tokenizer)
    
    for i, example in enumerate(mlm_examples, 1):
        print(f"Example {i}:")
        print(f"Input: {example}")
        
        predictions = predict_masked_tokens(model, tokenizer, example, device, top_k=3)
        
        for pred in predictions:
            print(f"\nTop 3 predictions for [MASK] at position {pred['position']}:")
            for token, prob in pred['predictions']:
                print(f"  {token}: {prob:.4f}")
        print()
    
    # Next Sentence Prediction Examples
    print("\n--- Next Sentence Prediction Examples ---\n")
    nsp_examples = create_nsp_examples()
    
    correct = 0
    for i, example in enumerate(nsp_examples, 1):
        sentence_a = example['sentence_a']
        sentence_b = example['sentence_b']
        true_label = example['label']
        
        pred, confidence, probs = evaluate_nsp(
            model, sentence_a, sentence_b, tokenizer, device
        )
        
        correct += (pred == true_label)
        
        print(f"Example {i}:")
        print(f"Sentence A: {sentence_a}")
        print(f"Sentence B: {sentence_b}")
        print(f"Prediction: {'Next Sentence' if pred == 1 else 'Not Next Sentence'}")
        print(f"Confidence: {confidence:.4f}")
        print(f"True Label: {'Next Sentence' if true_label == 1 else 'Not Next Sentence'}")
        print(f"✓ Correct" if pred == true_label else "✗ Incorrect")
        print()
    
    accuracy = 100.0 * correct / len(nsp_examples)
    print(f"NSP Accuracy on examples: {accuracy:.2f}%")
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Train Mini BERT')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden size (default: 256)')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of transformer layers (default: 4)')
    parser.add_argument('--num_attention_heads', type=int, default=4,
                        help='Number of attention heads (default: 4)')
    parser.add_argument('--intermediate_size', type=int, default=1024,
                        help='Intermediate size in FFN (default: 1024)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability (default: 0.1)')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of epochs (default: 3)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Max sequence length (default: 128)')
    parser.add_argument('--mlm_probability', type=float, default=0.15,
                        help='MLM masking probability (default: 0.15)')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation on pretrained model')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for evaluation')
    
    args = parser.parse_args()
    
    if args.eval_only and args.checkpoint:
        # Load pretrained model and evaluate
        device = get_device()
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        model = MiniBERT(
            vocab_size=len(tokenizer),
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size,
            max_position_embeddings=args.max_length,
            dropout=args.dropout
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        evaluate_model(model, tokenizer, device)
    else:
        # Train and evaluate
        model, tokenizer, device = train_model(args)
        evaluate_model(model, tokenizer, device)


if __name__ == '__main__':
    main()