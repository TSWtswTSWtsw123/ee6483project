"""
Example usage of the deep learning models
Demonstrates how to use trained models for inference
"""

import torch
import json
from data_utils import TextPreprocessor, Vocabulary, SentimentDataset
from deep_learning_models import (
    CNNClassifier, BiLSTMClassifier, AttentionBiLSTMClassifier
)


def example_preprocessing():
    """
    Example: Text preprocessing pipeline
    """
    print("="*60)
    print("Example 1: Text Preprocessing")
    print("="*60)

    sample_reviews = [
        "This product is absolutely amazing! Highly recommended!",
        "Terrible quality. Very disappointed with my purchase.",
        "It's okay, nothing special but does what it says.",
    ]

    print("\nOriginal Reviews:")
    for i, review in enumerate(sample_reviews, 1):
        print(f"{i}. {review}")

    print("\nAfter Preprocessing:")
    for i, review in enumerate(sample_reviews, 1):
        tokens = TextPreprocessor.preprocess(review)
        print(f"{i}. {' '.join(tokens)}")


def example_vocabulary():
    """
    Example: Vocabulary building
    """
    print("\n" + "="*60)
    print("Example 2: Vocabulary Building")
    print("="*60)

    # Sample tokenized texts
    sample_tokens = [
        ["excellent", "product", "highly", "recommend"],
        ["terrible", "quality", "waste", "money"],
        ["good", "value", "for", "price"],
    ]

    # Build vocabulary
    vocab = Vocabulary(min_freq=1)
    vocab.build(sample_tokens)

    print(f"\nVocabulary size: {len(vocab)}")
    print(f"\nWord to Index (first 10):")
    for word, idx in list(vocab.word2idx.items())[:10]:
        print(f"  {word}: {idx}")

    # Test encoding
    test_text = ["excellent", "product", "unknown_word"]
    encoded = vocab.encode(test_text)
    print(f"\nTest encoding:")
    print(f"  Original: {test_text}")
    print(f"  Encoded: {encoded}")
    print(f"  Decoded: {vocab.decode(encoded)}")


def example_model_inference():
    """
    Example: Model inference with dummy data
    """
    print("\n" + "="*60)
    print("Example 3: Model Inference")
    print("="*60)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 2
    SEQ_LEN = 200
    VOCAB_SIZE = 1000

    # Create dummy input
    dummy_text = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    dummy_lengths = torch.tensor([150, 180])

    print(f"\nInput shape: {dummy_text.shape}")
    print(f"Sequence lengths: {dummy_lengths.tolist()}")
    print(f"Device: {DEVICE}")

    # Test CNN
    print("\n1. CNN Classifier:")
    cnn_model = CNNClassifier(vocab_size=VOCAB_SIZE).to(DEVICE)
    dummy_text_gpu = dummy_text.to(DEVICE)
    with torch.no_grad():
        cnn_output = cnn_model(dummy_text_gpu)
    print(f"   Output shape: {cnn_output.shape}")
    print(f"   Sample predictions: {cnn_output[0].item():.4f}, {cnn_output[1].item():.4f}")

    # Test BiLSTM
    print("\n2. BiLSTM Classifier:")
    bilstm_model = BiLSTMClassifier(vocab_size=VOCAB_SIZE).to(DEVICE)
    dummy_lengths_gpu = dummy_lengths.to(DEVICE)
    with torch.no_grad():
        bilstm_output = bilstm_model(dummy_text_gpu, dummy_lengths_gpu)
    print(f"   Output shape: {bilstm_output.shape}")
    print(f"   Sample predictions: {bilstm_output[0].item():.4f}, {bilstm_output[1].item():.4f}")

    # Test Attention-BiLSTM
    print("\n3. Attention-BiLSTM Classifier:")
    attn_model = AttentionBiLSTMClassifier(vocab_size=VOCAB_SIZE).to(DEVICE)
    with torch.no_grad():
        attn_output, attn_weights = attn_model(dummy_text_gpu, dummy_lengths_gpu)
    print(f"   Output shape: {attn_output.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    print(f"   Sample predictions: {attn_output[0].item():.4f}, {attn_output[1].item():.4f}")
    print(f"   Top attention weights (sample 1):")
    top_indices = torch.argsort(attn_weights[0], descending=True)[:5]
    for idx in top_indices:
        print(f"     Position {idx.item()}: {attn_weights[0, idx].item():.4f}")


def example_model_comparison():
    """
    Example: Compare model sizes and inference speed
    """
    print("\n" + "="*60)
    print("Example 4: Model Comparison")
    print("="*60)

    VOCAB_SIZE = 1000
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    models = {
        'CNN': CNNClassifier(vocab_size=VOCAB_SIZE),
        'BiLSTM': BiLSTMClassifier(vocab_size=VOCAB_SIZE),
        'Attention-BiLSTM': AttentionBiLSTMClassifier(vocab_size=VOCAB_SIZE),
    }

    print(f"\nModel Comparison (Vocabulary size: {VOCAB_SIZE}):\n")
    print(f"{'Model':<20} {'Parameters':<15} {'Memory (MB)':<15}")
    print("-" * 50)

    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        # Rough memory estimate (4 bytes per float32)
        memory_mb = (params * 4) / (1024 ** 2)
        print(f"{name:<20} {params:<15,} {memory_mb:<15.2f}")

    # Inference speed comparison
    print("\n" + "="*60)
    print("Inference Time Comparison (100 batches, batch_size=64)")
    print("="*60)

    import time

    dummy_text = torch.randint(0, VOCAB_SIZE, (64, 200)).to(DEVICE)
    dummy_lengths = torch.randint(150, 200, (64,)).to(DEVICE)

    for name, model in models.items():
        model = model.to(DEVICE)
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                if name == 'Attention-BiLSTM':
                    _, _ = model(dummy_text, dummy_lengths)
                else:
                    _ = model(dummy_text, dummy_lengths)

        # Timing
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                if name == 'Attention-BiLSTM':
                    _, _ = model(dummy_text, dummy_lengths)
                else:
                    _ = model(dummy_text, dummy_lengths)
        elapsed = time.time() - start

        avg_time = (elapsed / 100) * 1000  # Convert to ms
        print(f"{name:<20}: {avg_time:.2f} ms per batch ({avg_time/64:.3f} ms per sample)")


def main():
    """
    Run all examples
    """
    print("\n")
    print("█" * 60)
    print("Deep Learning Models - Usage Examples")
    print("█" * 60)

    example_preprocessing()
    example_vocabulary()
    example_model_inference()
    example_model_comparison()

    print("\n" + "█" * 60)
    print("Examples completed!")
    print("█" * 60 + "\n")


if __name__ == '__main__':
    main()
