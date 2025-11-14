"""
Prediction script for generating submission.csv
Uses the best trained Attention-BiLSTM model
"""

import torch
import json
import csv
import numpy as np
from data_utils import create_dataloaders, TextPreprocessor, Vocabulary
from deep_learning_models import AttentionBiLSTMClassifier

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def generate_submission(model_path='best_attention_bilstm_model.pt',
                       test_file='test.json',
                       output_file='submission.csv',
                       device='cuda'):
    """
    Generate submission CSV file with predictions

    Args:
        model_path: path to saved model weights
        test_file: path to test JSON file
        output_file: path to output CSV file
        device: device to use ('cuda' or 'cpu')
    """
    print(f"Loading model from {model_path}...")

    # Load data to get vocabulary
    _, _, test_loader, vocab = create_dataloaders(
        train_file='train.json',
        test_file=test_file,
        batch_size=64,
        max_len=200,
        val_split=0.15
    )

    vocab_size = len(vocab)

    # Create model
    model = AttentionBiLSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=300,
        hidden_dim=256,
        num_layers=2,
        dropout_rate=0.5
    )

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"Model loaded. Making predictions...")

    # Get predictions
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for batch in test_loader:
            texts, lengths, _ = batch
            texts = texts.to(device)
            lengths = lengths.to(device)

            predictions, attention_weights = model(texts, lengths)
            all_predictions.extend((predictions > 0.5).cpu().numpy().flatten().tolist())
            all_probabilities.extend(predictions.cpu().numpy().flatten().tolist())

    # Save to CSV
    print(f"Saving predictions to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Prediction'])  # Header
        for idx, pred in enumerate(all_predictions):
            writer.writerow([idx, int(pred)])

    print(f"Submission file saved: {output_file}")
    print(f"Total predictions: {len(all_predictions)}")
    print(f"Positive predictions: {sum(all_predictions)}")
    print(f"Negative predictions: {len(all_predictions) - sum(all_predictions)}")
    print(f"Positive ratio: {sum(all_predictions) / len(all_predictions) * 100:.2f}%")

    return all_predictions, all_probabilities


def generate_ensemble_submission(model_paths, test_file='test.json',
                                output_file='submission_ensemble.csv',
                                device='cuda'):
    """
    Generate submission using ensemble of models

    Args:
        model_paths: dict of model names to model paths
        test_file: path to test JSON file
        output_file: path to output CSV file
        device: device to use
    """
    print("Loading ensemble models...")

    # Load data
    _, _, test_loader, vocab = create_dataloaders(
        train_file='train.json',
        test_file=test_file,
        batch_size=64,
        max_len=200,
        val_split=0.15
    )

    vocab_size = len(vocab)

    # Load all models
    models = {}
    for model_name, model_path in model_paths.items():
        if model_name == 'Attention-BiLSTM':
            model = AttentionBiLSTMClassifier(
                vocab_size=vocab_size,
                embedding_dim=300,
                hidden_dim=256,
                num_layers=2,
                dropout_rate=0.5
            )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        models[model_name] = model
        print(f"  Loaded {model_name}")

    print("Making ensemble predictions...")

    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for batch in test_loader:
            texts, lengths, _ = batch
            texts = texts.to(device)
            lengths = lengths.to(device)

            # Collect predictions from all models
            batch_probs = []
            for model_name, model in models.items():
                if model_name == 'Attention-BiLSTM':
                    pred, _ = model(texts, lengths)
                else:
                    pred = model(texts, lengths)
                batch_probs.append(pred.cpu().numpy().flatten())

            # Average predictions
            avg_prob = np.mean(batch_probs, axis=0)
            all_probabilities.extend(avg_prob)
            all_predictions.extend((avg_prob > 0.5).astype(int).tolist())

    # Save to CSV
    print(f"Saving ensemble predictions to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Prediction'])
        for idx, pred in enumerate(all_predictions):
            writer.writerow([idx, int(pred)])

    print(f"Submission file saved: {output_file}")
    print(f"Total predictions: {len(all_predictions)}")
    print(f"Positive predictions: {sum(all_predictions)}")
    print(f"Negative predictions: {len(all_predictions) - sum(all_predictions)}")
    print(f"Positive ratio: {sum(all_predictions) / len(all_predictions) * 100:.2f}%")

    return all_predictions, all_probabilities


def main():
    """
    Main prediction script
    """
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    print()

    # Generate submission using best model (Attention-BiLSTM)
    print("="*60)
    print("Generating submission using Attention-BiLSTM")
    print("="*60)
    predictions, probabilities = generate_submission(
        model_path='best_attention_bilstm_model.pt',
        test_file='test.json',
        output_file='submission.csv',
        device=DEVICE
    )

    print("\n" + "="*60)
    print("Submission generation complete!")
    print("="*60)


if __name__ == '__main__':
    main()
