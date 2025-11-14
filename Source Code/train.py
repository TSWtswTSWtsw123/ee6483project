"""
Training script for sentiment analysis models
Trains CNN, BiLSTM, and Attention-BiLSTM models with hyperparameter optimization
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import os
from tqdm import tqdm
from data_utils import create_dataloaders
from deep_learning_models import (
    CNNClassifier, BiLSTMClassifier, AttentionBiLSTMClassifier, WeightedBCELoss
)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class Trainer:
    """
    Trainer class for model training and evaluation
    """

    def __init__(self, model, train_loader, val_loader, test_loader=None,
                 device='cuda', model_name='model'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.model_name = model_name

        # Calculate class weights for weighted loss
        # Count positive and negative samples
        pos_count = sum([sum(batch[2]).item() for batch in train_loader])
        neg_count = sum([len(batch[2]) - sum(batch[2]).item() for batch in train_loader])
        pos_weight = float(neg_count) / float(pos_count)

        print(f"Class weights - Positive: 1.0, Negative: {pos_weight:.2f}")

        # Loss and optimizer
        self.criterion = WeightedBCELoss(pos_weight=pos_weight)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=True
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }

    def train_epoch(self):
        """
        Train one epoch
        Returns:
            avg_loss: average training loss
            accuracy: training accuracy
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        progress_bar = tqdm(self.train_loader, desc=f"{self.model_name} Training")
        for batch in progress_bar:
            texts, lengths, labels = batch
            texts = texts.to(self.device)
            lengths = lengths.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.model_name == 'Attention-BiLSTM':
                predictions, _ = self.model(texts, lengths)
            elif self.model_name == 'CNN':
                predictions = self.model(texts)
            else:  # BiLSTM
                predictions = self.model(texts, lengths)

            loss = self.criterion(predictions, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            # Store predictions and labels for accuracy calculation
            all_preds.extend((predictions > 0.5).cpu().detach().numpy().flatten().tolist())
            all_labels.extend(labels.cpu().detach().numpy().flatten().tolist())

            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def evaluate(self, data_loader=None):
        """
        Evaluate model on validation or test set
        Args:
            data_loader: DataLoader to evaluate on (default: validation set)
        Returns:
            loss: evaluation loss
            accuracy: evaluation accuracy
            precision: precision score
            recall: recall score
            f1: F1 score
            all_predictions: all predictions for this batch
            all_labels: all true labels
        """
        if data_loader is None:
            data_loader = self.val_loader

        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"{self.model_name} Evaluating"):
                texts, lengths, labels = batch
                texts = texts.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                if self.model_name == 'Attention-BiLSTM':
                    predictions, _ = self.model(texts, lengths)
                elif self.model_name == 'CNN':
                    predictions = self.model(texts)
                else:  # BiLSTM
                    predictions = self.model(texts, lengths)

                loss = self.criterion(predictions, labels)
                total_loss += loss.item()

                all_preds.extend((predictions > 0.5).cpu().numpy().flatten().tolist())
                all_labels.extend(labels.cpu().numpy().flatten().tolist())
                all_probs.extend(predictions.cpu().numpy().flatten().tolist())

        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels

    def train(self, num_epochs=15, patience=5):
        """
        Train model with early stopping
        Args:
            num_epochs: maximum number of epochs
            patience: early stopping patience
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = self.evaluate()

            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
            print(f"  Learning Rate: {current_lr:.2e}")
            print()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(
                    self.model.state_dict(),
                    f'best_{self.model_name.lower().replace("-", "_")}_model.pt'
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.model.load_state_dict(torch.load(f'best_{self.model_name.lower().replace("-", "_")}_model.pt'))
        print(f"Loaded best model with val_loss: {best_val_loss:.4f}")

    def get_predictions(self, return_probs=False):
        """
        Get predictions on test set
        Args:
            return_probs: whether to return probability scores
        Returns:
            predictions: list of predictions
            probabilities: list of probability scores (if return_probs=True)
        """
        if self.test_loader is None:
            raise ValueError("Test loader not provided")

        self.model.eval()
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"{self.model_name} Predicting"):
                texts, lengths, _ = batch
                texts = texts.to(self.device)
                lengths = lengths.to(self.device)

                if self.model_name == 'Attention-BiLSTM':
                    predictions, _ = self.model(texts, lengths)
                elif self.model_name == 'CNN':
                    predictions = self.model(texts)
                else:  # BiLSTM
                    predictions = self.model(texts, lengths)

                all_preds.extend((predictions > 0.5).cpu().numpy().flatten().tolist())
                all_probs.extend(predictions.cpu().numpy().flatten().tolist())

        if return_probs:
            return all_preds, all_probs
        return all_preds


def main():
    """
    Main training script
    """
    # Configuration
    BATCH_SIZE = 64
    MAX_LEN = 200
    NUM_EPOCHS = 15
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.5

    print(f"Using device: {DEVICE}")
    print()

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, vocab = create_dataloaders(
        train_file='train.json',
        test_file='test.json',
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
        val_split=0.15
    )
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    print()

    # Train models
    models_config = {
        'CNN': {
            'class': CNNClassifier,
            'params': {
                'vocab_size': vocab_size,
                'embedding_dim': EMBEDDING_DIM,
                'num_filters': 100,
                'filter_sizes': [3, 4, 5],
                'dropout_rate': DROPOUT_RATE
            }
        },
        'BiLSTM': {
            'class': BiLSTMClassifier,
            'params': {
                'vocab_size': vocab_size,
                'embedding_dim': EMBEDDING_DIM,
                'hidden_dim': HIDDEN_DIM,
                'num_layers': NUM_LAYERS,
                'dropout_rate': DROPOUT_RATE
            }
        },
        'Attention-BiLSTM': {
            'class': AttentionBiLSTMClassifier,
            'params': {
                'vocab_size': vocab_size,
                'embedding_dim': EMBEDDING_DIM,
                'hidden_dim': HIDDEN_DIM,
                'num_layers': NUM_LAYERS,
                'dropout_rate': DROPOUT_RATE
            }
        }
    }

    results = {}

    for model_name, config in models_config.items():
        print(f"{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")

        # Create model
        model = config['class'](**config['params'])
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print()

        # Create trainer
        trainer = Trainer(
            model,
            train_loader,
            val_loader,
            test_loader,
            device=DEVICE,
            model_name=model_name
        )

        # Train
        trainer.train(num_epochs=NUM_EPOCHS, patience=5)
        print()

        # Evaluate on validation set
        val_loss, val_acc, val_precision, val_recall, val_f1, val_preds, val_labels = trainer.evaluate()
        print(f"\nValidation Results for {model_name}:")
        print(f"  Accuracy: {val_acc:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  Recall: {val_recall:.4f}")
        print(f"  F1-Score: {val_f1:.4f}")

        # Get test predictions
        test_preds = trainer.get_predictions(return_probs=False)

        results[model_name] = {
            'val_accuracy': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'test_predictions': test_preds,
            'model': trainer.model
        }

        print()

    # Save results
    with open('training_results.json', 'w') as f:
        results_to_save = {
            model_name: {
                'val_accuracy': results[model_name]['val_accuracy'],
                'val_precision': results[model_name]['val_precision'],
                'val_recall': results[model_name]['val_recall'],
                'val_f1': results[model_name]['val_f1']
            }
            for model_name in results
        }
        json.dump(results_to_save, f, indent=4)

    print("\nTraining completed!")
    print(f"\nSummary of Results:")
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 68)
    for model_name, result in results.items():
        print(f"{model_name:<20} {result['val_accuracy']:<12.4f} {result['val_precision']:<12.4f} {result['val_recall']:<12.4f} {result['val_f1']:<12.4f}")

    return results


if __name__ == '__main__':
    results = main()
