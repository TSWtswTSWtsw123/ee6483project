"""
Deep Learning Models for Sentiment Analysis
Implements CNN, BiLSTM, and Attention-BiLSTM models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CNNClassifier(nn.Module):
    """
    CNN Classifier using multiple filter sizes for n-gram feature extraction
    Architecture: Embedding -> Multiple Conv1D -> Max Pooling -> FC -> Output
    """

    def __init__(self, vocab_size, embedding_dim=300, num_filters=100,
                 filter_sizes=[3, 4, 5], dropout_rate=0.5, num_classes=1):
        super(CNNClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Parallel convolutions with different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, fs)
            for fs in filter_sizes
        ])

        # Fully connected layers
        self.fc1 = nn.Linear(num_filters * len(filter_sizes), 128)
        self.batch_norm = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)

        self.filter_sizes = filter_sizes

    def forward(self, text):
        """
        Args:
            text: (batch_size, seq_len)
        Returns:
            output: (batch_size, 1)
        """
        # Embedding: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(text)

        # Transpose for conv1d: (batch_size, embedding_dim, seq_len)
        embedded = embedded.transpose(1, 2)

        # Apply convolutions and pooling
        conv_outputs = []
        for conv in self.convs:
            # Conv: (batch_size, num_filters, seq_len - filter_size + 1)
            conv_out = F.relu(conv(embedded))
            # Max pooling: (batch_size, num_filters)
            pooled = F.max_pool1d(conv_out, conv_out.shape[2])
            conv_outputs.append(pooled.squeeze(2))

        # Concatenate all pooled features: (batch_size, num_filters * len(filter_sizes))
        concat = torch.cat(conv_outputs, dim=1)

        # Fully connected layers
        hidden = self.dropout(F.relu(self.batch_norm(self.fc1(concat))))
        output = torch.sigmoid(self.fc2(hidden))

        return output


class BiLSTMClassifier(nn.Module):
    """
    BiLSTM Classifier for sentiment analysis
    Architecture: Embedding -> BiLSTM -> Last Hidden State -> FC -> Output
    """

    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256,
                 num_layers=2, dropout_rate=0.5, num_classes=1):
        super(BiLSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        # Fully connected layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)  # *2 for bidirectional
        self.batch_norm = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, text, lengths):
        """
        Args:
            text: (batch_size, seq_len)
            lengths: (batch_size,) actual lengths before padding
        Returns:
            output: (batch_size, 1)
        """
        # Embedding: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)

        # Pack padded sequence
        packed_embedded = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM: output (batch_size, seq_len, hidden_dim*2)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # Use last hidden state from both directions
        # hidden: (num_layers*2, batch_size, hidden_dim)
        # Take the last layer: (2, batch_size, hidden_dim)
        last_hidden = hidden[-2:, :, :].transpose(0, 1).contiguous()
        last_hidden = last_hidden.view(last_hidden.shape[0], -1)  # (batch_size, hidden_dim*2)

        # Fully connected layers
        hidden_state = self.dropout(F.relu(self.batch_norm(self.fc1(last_hidden))))
        output = torch.sigmoid(self.fc2(hidden_state))

        return output


class AttentionLayer(nn.Module):
    """
    Attention mechanism for BiLSTM
    Computes attention weights over all time steps
    """

    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch_size, seq_len, hidden_dim)
        Returns:
            context_vector: (batch_size, hidden_dim)
            attention_weights: (batch_size, seq_len)
        """
        # Compute attention scores: (batch_size, seq_len, 1)
        attention_scores = self.attention(lstm_output)

        # Apply softmax: (batch_size, seq_len)
        attention_weights = F.softmax(attention_scores.squeeze(2), dim=1)

        # Compute context vector as weighted sum: (batch_size, hidden_dim)
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),
            lstm_output
        ).squeeze(1)

        return context_vector, attention_weights


class AttentionBiLSTMClassifier(nn.Module):
    """
    Attention-enhanced BiLSTM Classifier (Primary Model)
    Architecture: Embedding -> BiLSTM -> Attention -> FC -> Output

    The attention mechanism helps the model focus on sentiment-bearing words
    """

    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256,
                 num_layers=2, dropout_rate=0.5, num_classes=1):
        super(AttentionBiLSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        # Attention layer
        self.attention = AttentionLayer(hidden_dim * 2)  # *2 for bidirectional

        # Fully connected layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.batch_norm = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, text, lengths):
        """
        Args:
            text: (batch_size, seq_len)
            lengths: (batch_size,) actual lengths before padding
        Returns:
            output: (batch_size, 1)
            attention_weights: (batch_size, seq_len) for interpretability
        """
        # Embedding
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)

        # Pack padded sequence
        packed_embedded = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # Unpack sequence
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Apply attention
        context_vector, attention_weights = self.attention(lstm_output)

        # Fully connected layers
        hidden_state = self.dropout(F.relu(self.batch_norm(self.fc1(context_vector))))
        output = torch.sigmoid(self.fc2(hidden_state))

        return output, attention_weights


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss to handle class imbalance
    Applies higher weights to minority class (negative reviews)
    """

    def __init__(self, pos_weight=1.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch_size, 1) predicted probabilities [0, 1]
            targets: (batch_size, 1) ground truth labels {0, 1}
        Returns:
            loss: scalar loss value
        """
        # pos_weight: weight for positive class
        # We use it to down-weight positive class and up-weight negative class
        neg_weight = self.pos_weight
        pos_weight = 1.0

        # Binary cross-entropy with weights
        bce = F.binary_cross_entropy(
            predictions,
            targets.float(),
            reduction='none'
        )

        # Apply weights based on target label
        weighted_loss = torch.where(
            targets == 1,
            pos_weight * bce,
            neg_weight * bce
        )

        return weighted_loss.mean()
