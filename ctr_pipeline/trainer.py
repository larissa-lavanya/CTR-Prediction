"""Training loop with periodic validation, early stopping and best-checkpoint restoration."""

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from ctr_pipeline.metrics import auc, logloss
from ctr_pipeline.models import embedding_l2_penalty, split_parameters


@dataclass
class TrainResult:
    best_val_logloss: float
    best_step: int
    steps: int
    train_seconds: float
    history: list = field(default_factory=list)


def predict_proba(model, data, batch_size=32768, device="cpu"):
    """Click probabilities for every row of `data`."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, data.n, batch_size):
            sparse = torch.from_numpy(data.sparse[start : start + batch_size]).to(device)
            dense = torch.from_numpy(data.dense[start : start + batch_size]).to(device)
            outputs.append(torch.sigmoid(model(sparse, dense)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def fit(
    model,
    train,
    val,
    epochs=2,
    batch_size=4096,
    lr=1e-3,
    emb_lr=1e-2,
    emb_l2=0.0,
    evals_per_epoch=2,
    patience=3,
    device="cpu",
    seed=0,
    log=print,
):
    """Train with mini-batch Adam; stop when validation log loss has not improved for `patience` checks.

    emb_l2 is the strength of the (lazy) L2 penalty on the embeddings used in each batch.
    """
    model.to(device)
    torch.manual_seed(seed)

    # Embedding tables get sparse gradients (SparseAdam); all other weights use Adam
    sparse_params, dense_params = split_parameters(model)
    optimizers = [torch.optim.SparseAdam(sparse_params, lr=emb_lr)]
    if dense_params:
        optimizers.append(torch.optim.Adam(dense_params, lr=lr))

    # Training data as tensors; each batch is gathered by a random permutation of the rows
    sparse_all = torch.from_numpy(train.sparse)
    dense_all = torch.from_numpy(train.dense)
    label_all = torch.from_numpy(train.label)

    steps_per_epoch = int(np.ceil(train.n / batch_size))
    eval_every = max(1, steps_per_epoch // evals_per_epoch)

    best = {"logloss": np.inf, "step": 0, "state": None}
    history, bad_checks, step, running_loss, running_n = [], 0, 0, 0.0, 0
    started = time.perf_counter()

    for epoch in range(epochs):
        order = torch.randperm(train.n)
        for start in range(0, train.n, batch_size):
            idx = order[start : start + batch_size]
            model.train()

            logits = model(sparse_all[idx].to(device), dense_all[idx].to(device))
            loss = F.binary_cross_entropy_with_logits(logits, label_all[idx].to(device))
            objective = loss + emb_l2 * embedding_l2_penalty(model) if emb_l2 > 0 else loss

            for optimizer in optimizers:
                optimizer.zero_grad()
            objective.backward()
            for optimizer in optimizers:
                optimizer.step()

            step += 1
            running_loss += loss.item() * len(idx)
            running_n += len(idx)

            # Periodic validation (early stopping on log loss)
            if step % eval_every == 0:
                p_val = predict_proba(model, val, device=device)
                val_logloss = logloss(val.label, p_val)
                history.append(
                    {
                        "step": step,
                        "epoch": step / steps_per_epoch,
                        "train_loss": running_loss / running_n,
                        "val_logloss": val_logloss,
                        "val_auc": auc(val.label, p_val),
                    }
                )
                log(
                    f"  step {step:6d} (epoch {step / steps_per_epoch:4.2f})  "
                    f"train loss {history[-1]['train_loss']:.4f}  val logloss {val_logloss:.4f}  val auc {history[-1]['val_auc']:.4f}"
                )
                running_loss, running_n = 0.0, 0

                if val_logloss < best["logloss"] - 1e-5:
                    best = {"logloss": val_logloss, "step": step, "state": copy.deepcopy(model.state_dict())}
                    bad_checks = 0
                else:
                    bad_checks += 1
                    if bad_checks >= patience:
                        log(f"  early stopping at step {step} (best was step {best['step']})")
                        break
        if bad_checks >= patience:
            break

    # Restore the weights from the best validation check
    if best["state"] is not None:
        model.load_state_dict(best["state"])

    return TrainResult(best["logloss"], best["step"], step, time.perf_counter() - started, history)


def time_calls(fn, n_iters=50, n_warmup=5):
    """Median wall-clock milliseconds of fn() over n_iters calls."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return 1000.0 * float(np.median(times))


def measure_model_latency(model, data, batch_sizes=(1, 1024)):
    """Inference latency on the CPU (hardware-neutral): ms per batch and microseconds per row."""
    model = model.to("cpu").eval()
    result = {}
    for size in batch_sizes:
        sparse = torch.from_numpy(data.sparse[:size])
        dense = torch.from_numpy(data.dense[:size])

        with torch.no_grad():
            ms = time_calls(lambda: model(sparse, dense))
        result[f"latency_ms_bs{size}"] = ms
        result[f"us_per_row_bs{size}"] = 1000.0 * ms / size
    return result
