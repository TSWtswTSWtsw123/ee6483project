"""
Data loading and preprocessing utilities for sentiment analysis
Handles JSON loading, tokenization, vocabulary building, and batching
"""

import json
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class Vocabulary:
    """
    Vocabulary class for handling word-to-index mappings
    Special tokens: <PAD> (index 0), <UNK> (index 1)
    """

    def __init__(self, min_freq=1):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = Counter()
        self.min_freq = min_freq

    def build(self, texts):
        """
        Build vocabulary from texts
        Args:
            texts: list of tokenized text (list of list of words)
        """
        # Count word frequencies
        for tokens in texts:
            self.word_freq.update(tokens)

        # Add words that appear at least min_freq times
        idx = 2
        for word, freq in self.word_freq.most_common():
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        print(f"Vocabulary size: {len(self.word2idx)}")

    def encode(self, tokens):
        """
        Convert tokens to indices
        Args:
            tokens: list of words
        Returns:
            list of indices
        """
        return [self.word2idx.get(token, 1) for token in tokens]  # 1 is UNK

    def decode(self, indices):
        """
        Convert indices back to tokens
        Args:
            indices: list of indices
        Returns:
            list of words
        """
        return [self.idx2word.get(idx, '<UNK>') for idx in indices]

    def __len__(self):
        return len(self.word2idx)


class TextPreprocessor:
    """
    Text preprocessing utilities
    Handles cleaning, tokenization, and normalization
    """

    @staticmethod
    def clean_text(text):
        """
        Clean text by removing special characters and extra whitespace
        Args:
            text: raw text string
        Returns:
            cleaned text string
        """
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters but keep punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\!\?]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    @staticmethod
    def tokenize(text):
        """
        Tokenize text into words
        Args:
            text: text string
        Returns:
            list of tokens
        """
        tokens = word_tokenize(text)
        return tokens

    @staticmethod
    def preprocess(text):
        """
        Complete preprocessing pipeline
        Args:
            text: raw text string
        Returns:
            list of preprocessed tokens
        """
        text = TextPreprocessor.clean_text(text)
        tokens = TextPreprocessor.tokenize(text)
        return tokens


class SentimentDataset(Dataset):
    """
    PyTorch Dataset for sentiment analysis
    """

    def __init__(self, texts, labels, vocab, max_len=200):
        """
        Args:
            texts: list of text strings or list of token lists
            labels: list of sentiment labels (0 or 1)
            vocab: Vocabulary object
            max_len: maximum sequence length
        """
        self.vocab = vocab
        self.max_len = max_len

        # Preprocess and encode texts
        self.encoded_texts = []
        self.lengths = []

        for text in texts:
            if isinstance(text, str):
                tokens = TextPreprocessor.preprocess(text)
            else:
                tokens = text

            # Encode to indices
            indices = vocab.encode(tokens)

            # Truncate to max_len
            indices = indices[:max_len]

            # Store actual length and padded sequence
            self.lengths.append(len(indices))
            self.encoded_texts.append(indices)

        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns:
            text: encoded and padded text
            length: actual length before padding
            label: sentiment label
        """
        text = torch.tensor(self.encoded_texts[idx], dtype=torch.long)
        length = self.lengths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return text, length, label


def collate_batch(batch):
    """
    Custom collate function for DataLoader
    Pads sequences to same length and creates length tensor

    Args:
        batch: list of (text, length, label) tuples
    Returns:
        texts: padded tensor (batch_size, seq_len)
        lengths: tensor of actual lengths (batch_size,)
        labels: tensor of labels (batch_size, 1)
    """
    texts, lengths, labels = zip(*batch)

    # Get maximum length in batch
    max_len = max(lengths)

    # Pad texts to max_len
    padded_texts = []
    for text in texts:
        if len(text) < max_len:
            padding = torch.zeros(max_len - len(text), dtype=torch.long)
            text = torch.cat([text, padding])
        padded_texts.append(text)

    texts = torch.stack(padded_texts)
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.stack([torch.tensor([label]) for label in labels])

    return texts, lengths, labels


def load_json_data(file_path):
    """
    Load sentiment data from JSON file
    Expected format: [{"reviews": "...", "sentiments": 0/1}, ...]
    For test files: [{"reviews": "..."}, ...]

    Args:
        file_path: path to JSON file
    Returns:
        texts: list of review texts
        labels: list of sentiment labels (None for test files)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = [item['reviews'] for item in data]

    # Check if labels exist (not present in test file)
    if data and 'sentiments' in data[0]:
        labels = [item['sentiments'] for item in data]
    else:
        labels = None

    return texts, labels


def create_dataloaders(train_file, test_file=None, val_split=0.15,
                       batch_size=64, max_len=200, num_workers=0):
    """
    Create DataLoaders for training and evaluation

    Args:
        train_file: path to training JSON file
        test_file: path to test JSON file (optional)
        val_split: validation split ratio (0.15 = 15%)
        batch_size: batch size for DataLoader
        max_len: maximum sequence length
        num_workers: number of workers for DataLoader
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        test_loader: DataLoader for testing (if test_file provided)
        vocab: Vocabulary object
    """
    # Load training data
    train_texts, train_labels = load_json_data(train_file)
    print(f"Loaded {len(train_texts)} training samples")

    # Split into train and validation
    num_samples = len(train_texts)
    num_val = int(num_samples * val_split)
    num_train = num_samples - num_val

    # Stratified split to maintain class balance
    pos_indices = [i for i in range(num_samples) if train_labels[i] == 1]
    neg_indices = [i for i in range(num_samples) if train_labels[i] == 0]

    np.random.seed(42)
    val_pos = np.random.choice(pos_indices, size=int(len(pos_indices) * val_split), replace=False)
    val_neg = np.random.choice(neg_indices, size=int(len(neg_indices) * val_split), replace=False)
    val_indices = list(val_pos) + list(val_neg)
    train_indices = [i for i in range(num_samples) if i not in val_indices]

    train_texts_split = [train_texts[i] for i in train_indices]
    train_labels_split = [train_labels[i] for i in train_indices]
    val_texts = [train_texts[i] for i in val_indices]
    val_labels = [train_labels[i] for i in val_indices]

    print(f"Training: {len(train_texts_split)}, Validation: {len(val_texts)}")
    print(f"Training - Positive: {sum(train_labels_split)}, Negative: {len(train_labels_split) - sum(train_labels_split)}")
    print(f"Validation - Positive: {sum(val_labels)}, Negative: {len(val_labels) - sum(val_labels)}")

    # Build vocabulary from training texts
    train_tokens = [TextPreprocessor.preprocess(text) for text in train_texts_split]
    vocab = Vocabulary(min_freq=1)
    vocab.build(train_tokens)

    # Create datasets
    train_dataset = SentimentDataset(train_texts_split, train_labels_split, vocab, max_len=max_len)
    val_dataset = SentimentDataset(val_texts, val_labels, vocab, max_len=max_len)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_workers
    )

    test_loader = None
    if test_file:
        test_texts, test_labels = load_json_data(test_file)
        print(f"Loaded {len(test_texts)} test samples")
        # Use dummy labels (0) for test data since they don't have true labels
        dummy_labels = [0] * len(test_texts)
        test_dataset = SentimentDataset(test_texts, dummy_labels, vocab, max_len=max_len)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers=num_workers
        )

    return train_loader, val_loader, test_loader, vocab
