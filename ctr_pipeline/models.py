"""CTR models: Logistic Regression, Factorization Machine, DeepFM and DCN-v2 (all return logits)."""

import numpy as np
import torch
from torch import nn


class FieldEmbeddings(nn.Module):
    """One big embedding table shared by all categorical fields (each field owns a slice of rows)."""

    def __init__(self, vocab_sizes, dim):
        super().__init__()
        offsets = np.concatenate([[0], np.cumsum(vocab_sizes)[:-1]])
        self.register_buffer("offsets", torch.as_tensor(offsets, dtype=torch.long))

        # Sparse gradients: only the rows used in a batch are updated (essential for huge tables)
        self.table = nn.Embedding(int(np.sum(vocab_sizes)), dim, sparse=True)
        nn.init.normal_(self.table.weight, std=0.01)

    def forward(self, sparse):
        """(batch, n_fields) indices -> (batch, n_fields, dim) vectors."""
        self.last_lookup = self.table(sparse.long() + self.offsets)  # kept for the L2 penalty
        return self.last_lookup


def make_mlp(in_dim, hidden, dropout):
    """Fully connected ReLU network; returns the module and its output width."""
    layers, width = [], in_dim
    for size in hidden:
        layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
        width = size
    return nn.Sequential(*layers), width


class LogisticRegression(nn.Module):
    """logit = bias + sum of one weight per categorical value + linear term of the dense features."""

    def __init__(self, vocab_sizes, n_dense, **unused):
        super().__init__()
        self.weights = FieldEmbeddings(vocab_sizes, 1)
        self.dense = nn.Linear(n_dense, 1)  # also holds the bias

    def forward(self, sparse, dense):
        return self.weights(sparse).sum(dim=(1, 2)) + self.dense(dense).squeeze(1)


class FactorizationMachine(nn.Module):
    """First-order terms plus ALL pairwise interactions via dot products of field embeddings."""

    def __init__(self, vocab_sizes, n_dense, emb_dim=16, **unused):
        super().__init__()
        self.first_order = LogisticRegression(vocab_sizes, n_dense)
        self.embeddings = FieldEmbeddings(vocab_sizes, emb_dim)

    @staticmethod
    def pairwise_interactions(emb):
        """sum_{i<j} <v_i, v_j> computed in O(fields * dim): 0.5 * ((sum v)^2 - sum v^2)."""
        square_of_sum = emb.sum(dim=1).pow(2)
        sum_of_square = emb.pow(2).sum(dim=1)
        return 0.5 * (square_of_sum - sum_of_square).sum(dim=1)

    def score(self, sparse, dense, emb):
        """FM logit given already looked-up embeddings (lets DeepFM share one lookup)."""
        return self.first_order(sparse, dense) + self.pairwise_interactions(emb)

    def forward(self, sparse, dense):
        return self.score(sparse, dense, self.embeddings(sparse))


class DeepFM(nn.Module):
    """FM (low-order interactions) + an MLP on the same embeddings (high-order interactions)."""

    def __init__(self, vocab_sizes, n_dense, emb_dim=16, hidden=(256, 128), dropout=0.1, **unused):
        super().__init__()
        self.fm = FactorizationMachine(vocab_sizes, n_dense, emb_dim)
        self.mlp, width = make_mlp(len(vocab_sizes) * emb_dim + n_dense, hidden, dropout)
        self.out = nn.Linear(width, 1)

    def forward(self, sparse, dense):
        emb = self.fm.embeddings(sparse)

        # The deep branch sees the flattened embeddings and the dense features
        deep_input = torch.cat([emb.flatten(start_dim=1), dense], dim=1)
        deep_logit = self.out(self.mlp(deep_input)).squeeze(1)

        return self.fm.score(sparse, dense, emb) + deep_logit


class CrossLayer(nn.Module):
    """DCN-v2 cross layer: x_next = x0 * (W x + b) + x  (explicit feature crosses)."""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x0, x):
        return x0 * self.linear(x) + x


class DCNv2(nn.Module):
    """Parallel DCN-v2: a stack of cross layers and an MLP on the same input, then a linear head."""

    def __init__(self, vocab_sizes, n_dense, emb_dim=16, hidden=(256, 128), dropout=0.1, n_cross=3, **unused):
        super().__init__()
        self.embeddings = FieldEmbeddings(vocab_sizes, emb_dim)
        input_dim = len(vocab_sizes) * emb_dim + n_dense

        self.cross_layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(n_cross)])
        self.mlp, width = make_mlp(input_dim, hidden, dropout)
        self.out = nn.Linear(width + (input_dim if n_cross > 0 else 0), 1)

    def forward(self, sparse, dense):
        x0 = torch.cat([self.embeddings(sparse).flatten(start_dim=1), dense], dim=1)

        # Cross branch (skipped when n_cross = 0, which leaves a plain deep network)
        features = [self.mlp(x0)]
        if len(self.cross_layers) > 0:
            x = x0
            for layer in self.cross_layers:
                x = layer(x0, x)
            features.append(x)

        return self.out(torch.cat(features, dim=1)).squeeze(1)


MODELS = {"lr": LogisticRegression, "fm": FactorizationMachine, "deepfm": DeepFM, "dcnv2": DCNv2}


def build_model(name, vocab_sizes, n_dense, **hyperparams):
    """Create a model by name; unknown hyperparameters are ignored by models that do not use them."""
    return MODELS[name](vocab_sizes, n_dense, **hyperparams)


def embedding_l2_penalty(model):
    """Mean over the batch of the squared norm of the embeddings that were just looked up.

    Only the rows used in this batch are penalized ("lazy" L2), which is cheap and works with
    sparse optimizers. Call after the forward pass.
    """
    penalty = 0.0
    for module in model.modules():
        if isinstance(module, FieldEmbeddings):
            penalty = penalty + module.last_lookup.pow(2).sum(dim=(1, 2)).mean()
    return penalty


def split_parameters(model):
    """Separate embedding tables (sparse gradients, SparseAdam) from everything else (Adam)."""
    sparse_params, dense_params = [], []
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            sparse_params += list(module.parameters(recurse=False))
        else:
            dense_params += list(module.parameters(recurse=False))
    return sparse_params, dense_params


def count_parameters(model):
    """Total parameters, and how many of them live in embedding tables."""
    sparse_params, dense_params = split_parameters(model)
    embedding = sum(p.numel() for p in sparse_params)
    other = sum(p.numel() for p in dense_params)
    return {"params_total": embedding + other, "params_embedding": embedding, "params_dense": other}
