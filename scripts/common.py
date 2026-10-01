"""Exact inference contract for the controlled matched BiLSTM arm."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text())
SEEDS = CONFIG["seeds"]
THRESHOLD = CONFIG["threshold"]


class BiLSTMIntentPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        c = CONFIG["model"]
        self.input_proj = nn.Sequential(nn.Linear(c["input_dim"], c["projection_dim"]), nn.ReLU())
        self.bilstm = nn.LSTM(input_size=c["projection_dim"], hidden_size=c["hidden_dim"],
                              num_layers=c["num_layers"], dropout=c["dropout"],
                              bidirectional=True, batch_first=True)
        self.head = nn.Linear(c["hidden_dim"] * 2, 1)

    def forward(self, x):
        out, _ = self.bilstm(self.input_proj(x))
        return self.head(out[:, -1, :])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ensemble(device="cpu"):
    models = []
    stats = []
    for seed in SEEDS:
        d = ROOT / "private" / "checkpoints" / f"seed{seed}"
        mean = np.load(d / "norm_mean.npy").astype(np.float32)
        std = np.load(d / "norm_std.npy").astype(np.float32)
        assert mean.shape == std.shape == (5,)
        assert np.all(np.isfinite(mean)) and np.all(np.isfinite(std)) and np.all(std > 0)
        model = BiLSTMIntentPredictor().to(device)
        checkpoint = torch.load(d / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint["state_dict"])
        model.eval()
        assert sum(p.numel() for p in model.parameters()) == 2237313
        models.append(model)
        stats.append((mean, std))
    for mean, std in stats[1:]:
        np.testing.assert_allclose(mean, stats[0][0], rtol=0, atol=1e-6)
        np.testing.assert_allclose(std, stats[0][1], rtol=0, atol=1e-6)
    return models, stats


@torch.inference_mode()
def predict_per_seed(models, stats, x, device="cpu", batch=256):
    assert x.ndim == 3 and x.shape[1:] == (16, 5), x.shape
    out = []
    for model, (mean, std) in zip(models, stats):
        p = []
        for i in range(0, len(x), batch):
            z = ((x[i:i + batch].astype(np.float32) - mean) / std).astype(np.float32)
            logits = model(torch.from_numpy(z).to(device)).squeeze(-1)
            p.append(torch.sigmoid(logits).cpu().numpy())
        out.append(np.concatenate(p))
    return np.stack(out)


def load_splits():
    import pickle
    d = ROOT / "private" / "data"
    x = np.load(d / "X.npy").astype(np.float32)
    y = np.load(d / "y.npy").astype(np.int8)
    meta = pickle.loads((d / "meta.pkl").read_bytes())
    assert x.shape == (4906, 16, 5) and y.shape == (4906,) and len(meta) == 4906
    sid = np.array([m["set_id"] for m in meta])
    tr = np.isin(sid, CONFIG["splits"]["train"])
    va = np.isin(sid, CONFIG["splits"]["validation"])
    te = np.isin(sid, CONFIG["splits"]["test"])
    assert (tr.sum(), va.sum(), te.sum()) == (2178, 634, 2094)
    return x[tr], y[tr], x[va], y[va], x[te], y[te], [m for m, take in zip(meta, te) if take]
