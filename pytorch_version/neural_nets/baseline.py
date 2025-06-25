from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Callable

@dataclass
class Params:
    seed: int = 42

    layer: str = "LSTM"  # "DNN" #"LSTM"
    activation: str = "tanh"  # "relu" "tanh"
    kernel_initializer: str = "glorot_uniform"  # "he_normal" "glorot_uniform"
    input_dim: int = None
    hidden_dim: int = 32
    n_hidden_layer: int = 2
    dropout: float = 0.1
    output_dim: int = 1

    lr: float = 0.0001
    l1: float = 0.
    opt: str = 'adam'
    criterion: str = 'mse'  # Changed to string for easier handling
    gradient_clip: float = None

    batch_size: int = 32
    n_steps: int = 100
    days_ahead: int = 100
    epochs: int = 100
    shuffle: bool = False
    patience_es: int = 5

    metric: str = 'mse'  # Changed to string
    metric_direction: str = 'min'  # 'min' or 'max'
    fold: int = None

    attention_heads: int = 0
    mask: bool = True

    method_aggregate_target: str = 'return'
    type_learn: str = 'regression'

    threshold: float = 0.9


class MCDropout(nn.Module):
    """Monte Carlo Dropout - always active during training and inference"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p > 0:
            return nn.functional.dropout(x, p=self.p, training=True)
        return x


class AttentionLayer(nn.Module):
    """Multi-head attention layer for sequences"""

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        return attn_output


class NN(nn.Module):
    """PyTorch Neural Network implementation"""

    def __init__(self, params: Params):
        super().__init__()
        self.params = params
        self.layers = nn.ModuleList()

        # Build the network
        if params.layer == 'DNN':
            self._build_dnn()
        elif params.layer in ['LSTM', 'GRU']:
            self._build_rnn()

    def _get_activation(self):
        if self.params.activation == 'relu':
            return nn.ReLU()
        elif self.params.activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif self.params.activation == 'tanh':
            return nn.Tanh()
        else:
            return nn.Identity()

    def _build_dnn(self):
        # Input projection
        self.layers.append(nn.Linear(self.params.input_dim, self.params.hidden_dim))
        self.layers.append(self._get_activation())

        if self.params.attention_heads > 0:
            self.layers.append(AttentionLayer(self.params.hidden_dim, self.params.attention_heads))

        self.layers.append(MCDropout(self.params.dropout))

        # Hidden layers
        for _ in range(self.params.n_hidden_layer):
            self.layers.append(nn.Linear(self.params.hidden_dim, self.params.hidden_dim))
            self.layers.append(self._get_activation())

            if self.params.attention_heads > 0:
                self.layers.append(AttentionLayer(self.params.hidden_dim, self.params.attention_heads))

            self.layers.append(MCDropout(self.params.dropout))

        # Output layer
        self.layers.append(nn.Linear(self.params.hidden_dim, self.params.output_dim))

    def _build_rnn(self):
        rnn_class = nn.LSTM if self.params.layer == 'LSTM' else nn.GRU

        # First RNN layer
        self.layers.append(
            rnn_class(self.params.input_dim, self.params.hidden_dim,
                      batch_first=True, dropout=0 if self.params.n_hidden_layer == 1 else self.params.dropout)
        )

        if self.params.attention_heads > 0:
            self.layers.append(AttentionLayer(self.params.hidden_dim, self.params.attention_heads))

        # Additional RNN layers
        for i in range(1, self.params.n_hidden_layer):
            self.layers.append(
                rnn_class(self.params.hidden_dim, self.params.hidden_dim,
                          batch_first=True, dropout=self.params.dropout if i < self.params.n_hidden_layer - 1 else 0)
            )

            if self.params.attention_heads > 0:
                self.layers.append(AttentionLayer(self.params.hidden_dim, self.params.attention_heads))

        # Final RNN layer without return_sequences
        self.final_rnn = rnn_class(self.params.hidden_dim, self.params.hidden_dim, batch_first=True)

        # Output layer
        self.output_layer = nn.Linear(self.params.hidden_dim, self.params.output_dim)
        self.dropout = MCDropout(self.params.dropout)

    def forward(self, x):
        if self.params.layer == 'DNN':
            # Flatten the time dimension for DNN
            x = x.squeeze(1)
            for layer in self.layers:
                x = layer(x)
            return x

        elif self.params.layer in ['LSTM', 'GRU']:
            # Process through RNN layers
            for layer in self.layers:
                if isinstance(layer, (nn.LSTM, nn.GRU)):
                    x, _ = layer(x)
                elif isinstance(layer, AttentionLayer):
                    x = layer(x)

            # Final RNN layer - take last timestep
            output, _ = self.final_rnn(x)
            x = output[:, -1, :]  # Take last timestep

            # Apply dropout and output layer
            x = self.dropout(x)
            x = self.output_layer(x)

            return x

    def get_optimizer(self):
        # Add L1 regularization if specified
        if self.params.l1 > 0:
            # We'll handle L1 regularization in the training loop
            pass

        if self.params.opt == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.params.lr)
        elif self.params.opt == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.params.lr)
        else:
            raise ValueError('Only Adam and SGD optimizers have been implemented')

        return optimizer

    def get_criterion(self):
        if self.params.criterion == 'mse':
            return nn.MSELoss()
        elif self.params.criterion == 'mae':
            return nn.L1Loss()
        elif self.params.criterion == 'bce':
            return nn.BCEWithLogitsLoss()
        else:
            return nn.MSELoss()  # Default

    def l1_regularization(self):
        """Calculate L1 regularization term"""
        l1_reg = 0
        for param in self.parameters():
            l1_reg += torch.sum(torch.abs(param))
        return self.params.l1 * l1_reg