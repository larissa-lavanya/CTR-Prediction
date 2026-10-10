"""Model shapes, the FM identity, parameter counts and a learning check for every model."""

import itertools

import numpy as np
import pytest
import torch

from ctr_pipeline.dataset import CTRData
from ctr_pipeline.metrics import auc
from ctr_pipeline.models import FactorizationMachine, build_model, count_parameters
from ctr_pipeline.trainer import fit, predict_proba

VOCAB_SIZES = [6, 5, 4]
N_DENSE = 2


def make_data(n=6000, seed=0):
    """Click depends on field 0 (a main effect) and on the PAIR of fields 1 and 2 (an interaction)."""
    rng = np.random.default_rng(seed)
    sparse = np.stack([rng.integers(0, v, n) for v in VOCAB_SIZES], axis=1).astype(np.int32)
    dense = rng.standard_normal((n, N_DENSE)).astype(np.float32)
    logit = 1.2 * (sparse[:, 0] % 2) + 1.5 * ((sparse[:, 1] + sparse[:, 2]) % 2) + 0.5 * dense[:, 0] - 1.5
    label = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.float32)
    return CTRData(sparse, dense, label, np.arange(n))


@pytest.mark.parametrize("name", ["lr", "fm", "deepfm", "dcnv2"])
def test_forward_shape(name):
    model = build_model(name, VOCAB_SIZES, N_DENSE, emb_dim=4)
    sparse = torch.randint(0, 4, (7, len(VOCAB_SIZES)))
    dense = torch.randn(7, N_DENSE)
    assert model(sparse, dense).shape == (7,)


def test_fm_pairwise_formula_matches_brute_force():
    emb = torch.randn(5, 4, 3)  # batch, fields, dim
    fast = FactorizationMachine.pairwise_interactions(emb)

    slow = torch.zeros(5)
    for i, j in itertools.combinations(range(4), 2):
        slow += (emb[:, i] * emb[:, j]).sum(dim=1)
    assert torch.allclose(fast, slow, atol=1e-5)


def test_parameter_counts():
    lr = build_model("lr", VOCAB_SIZES, N_DENSE)
    counts = count_parameters(lr)
    assert counts["params_embedding"] == sum(VOCAB_SIZES)
    assert counts["params_dense"] == N_DENSE + 1  # dense weights and bias

    fm = build_model("fm", VOCAB_SIZES, N_DENSE, emb_dim=8)
    assert count_parameters(fm)["params_embedding"] == sum(VOCAB_SIZES) * (1 + 8)


def test_dcn_without_cross_layers_still_works():
    model = build_model("dcnv2", VOCAB_SIZES, N_DENSE, emb_dim=4, n_cross=0)
    assert model(torch.zeros(3, len(VOCAB_SIZES), dtype=torch.long), torch.zeros(3, N_DENSE)).shape == (3,)


@pytest.mark.parametrize("name,min_auc", [("lr", 0.62), ("fm", 0.70), ("deepfm", 0.70), ("dcnv2", 0.70)])
def test_models_learn(name, min_auc):
    # Interaction-aware models should beat the logistic regression ceiling on this data
    data = make_data()
    train, val, test = data.head(4000), CTRData(data.sparse[4000:5000], data.dense[4000:5000], data.label[4000:5000], data.row_id[4000:5000]), \
        CTRData(data.sparse[5000:], data.dense[5000:], data.label[5000:], data.row_id[5000:])

    model = build_model(name, VOCAB_SIZES, N_DENSE, emb_dim=8, hidden=(32, 16), dropout=0.0)
    fit(model, train, val, epochs=25, batch_size=256, lr=5e-3, emb_lr=2e-2, patience=100, log=lambda s: None)

    assert auc(test.label, predict_proba(model, test)) > min_auc
