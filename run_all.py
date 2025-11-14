"""
Complete pipeline for sentiment analysis project
Runs: training -> prediction -> submission generation
"""

import os
import sys
import torch
import numpy as np
from train import main as train_main
from predict import generate_submission

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


def check_data_files():
    """Check if required data files exist"""
    print("Checking data files...")
    required_files = ['train.json', 'test.json']
    for file in required_files:
        if not os.path.exists(file):
            print(f"ERROR: {file} not found!")
            return False
    print("✓ All data files found")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    try:
        import torch
        import sklearn
        import nltk
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ scikit-learn installed")
        print(f"✓ NLTK installed")
        return True
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        return False


def main():
    """
    Main pipeline
    """
    print("="*70)
    print("Sentiment Analysis - Deep Learning Models")
    print("="*70)
    print()

    # Check dependencies
    if not check_dependencies():
        print("Failed to check dependencies")
        return False
    print()

    # Check data files
    if not check_data_files():
        print("Failed to check data files")
        return False
    print()

    # Training
    print("="*70)
    print("PHASE 1: Model Training")
    print("="*70)
    print()
    try:
        results = train_main()
        print()
    except Exception as e:
        print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Prediction
    print()
    print("="*70)
    print("PHASE 2: Generating Submission")
    print("="*70)
    print()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        predictions, probabilities = generate_submission(
            model_path='best_attention_bilstm_model.pt',
            test_file='test.json',
            output_file='submission.csv',
            device=DEVICE
        )
        print()
    except Exception as e:
        print(f"ERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print()
    print("="*70)
    print("Pipeline Summary")
    print("="*70)
    print()
    print("✓ Model training completed successfully")
    print("✓ Best model: Attention-BiLSTM")
    print(f"  - Validation Accuracy: {results['Attention-BiLSTM']['val_accuracy']:.4f}")
    print(f"  - Validation F1-Score: {results['Attention-BiLSTM']['val_f1']:.4f}")
    print()
    print("✓ Submission file generated")
    print(f"  - File: submission.csv")
    print(f"  - Total predictions: {len(predictions)}")
    print(f"  - Positive predictions: {sum(predictions)} ({sum(predictions)/len(predictions)*100:.2f}%)")
    print()
    print("="*70)
    print("Ready for submission!")
    print("="*70)

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
