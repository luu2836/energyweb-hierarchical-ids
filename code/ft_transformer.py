"""FT-Transformer deep IDS baseline (for the deep-baseline comparison).

Feature-tokenizer transformer over tabular flow features: each feature is
projected to a d_model token, a [CLS] token is prepended, and a small Transformer
encoder produces the class logits from the [CLS] embedding.

Usage (from code/):
    python ft_transformer.py unsw
    python ft_transformer.py cic
"""
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from config import CIC, UNSW, SEED
from baseline import prep, ID_COLS
from nn_hier import signed_log1p

torch.set_num_threads(8)
SEEDS = [1, 2, 3]
CAP = 40000          # cap training rows (deep baseline trained under a fixed CPU budget)
torch.manual_seed(SEED)
np.random.seed(SEED)


class FTTransformer(nn.Module):
    def __init__(self, d_feat, ncls, d=32, heads=2, layers=2, drop=0.1):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d_feat, d) * 0.02)
        self.b = nn.Parameter(torch.zeros(d_feat, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d,
                                         dropout=drop, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, ncls)

    def forward(self, x):
        tok = x.unsqueeze(-1) * self.w.unsqueeze(0) + self.b.unsqueeze(0)
        cls = self.cls.expand(x.size(0), -1, -1)
        h = torch.cat([cls, tok], dim=1)
        h = self.enc(h)
        return self.head(self.norm(h[:, 0]))


def train_ft(X, y, ncls, seed, d=32, layers=2, epochs=20, patience=4, batch=1024, lr=1e-3):
    torch.manual_seed(seed)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.15, stratify=y, random_state=seed)
    if len(Xtr) > CAP:
        Xtr, _, ytr, _ = train_test_split(Xtr, ytr, train_size=CAP, stratify=ytr, random_state=seed)
    xt = torch.tensor(Xtr); yt = torch.tensor(ytr, dtype=torch.long)
    xv = torch.tensor(Xva); yv = torch.tensor(yva, dtype=torch.long)
    crit = nn.CrossEntropyLoss()
    m = FTTransformer(X.shape[1], ncls, d=d, layers=layers)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    n = len(xt); best, wait, state = 1e9, 0, None
    for ep in range(epochs):
        t0 = time.perf_counter()
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]; opt.zero_grad()
            loss = crit(m(xt[idx]), yt[idx]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = crit(m(xv), yv).item()
        print("    epoch {:2d}  val_loss={:.4f}  ({:.0f}s)".format(ep, vl, time.perf_counter() - t0), flush=True)
        if vl < best - 1e-4:
            best, wait, state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    m.load_state_dict(state); m.eval()
    return m


def run(name, path, label_col, dedup):
    print("=" * 78); print(name, "| FT-Transformer | seeds", SEEDS); print("=" * 78)
    df = pd.read_csv(path, low_memory=False)
    if dedup:
        df = df.drop_duplicates(subset=[c for c in df.columns if c != label_col]).reset_index(drop=True)
    X, y = prep(df, label_col, ID_COLS)
    cls = sorted(y.unique())
    pm = {c: k for k, c in enumerate(cls)}
    res = []
    for s in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)
        a = signed_log1p(Xtr.values.astype(np.float64)); b = signed_log1p(Xte.values.astype(np.float64))
        mu, sd = a.mean(0), a.std(0); sd[sd == 0] = 1
        Xtr_s = ((a - mu) / sd).astype(np.float32); Xte_s = ((b - mu) / sd).astype(np.float32)
        ytr_i = np.array([pm[c] for c in ytr], dtype=np.int64)
        t0 = time.perf_counter()
        m = train_ft(Xtr_s, ytr_i, len(cls), s)
        with torch.no_grad():
            out = m(torch.tensor(Xte_s)).argmax(1).numpy()
        dt = time.perf_counter() - t0
        pred = np.array([cls[v] for v in out], dtype=object)
        macro = f1_score(yte, pred, average="macro", zero_division=0)
        per = f1_score(yte, pred, average=None, labels=cls, zero_division=0)
        res.append((accuracy_score(yte, pred), macro, float(np.min(per))))
        print("  seed {} done ({:.0f}s)  macro-F1={:.4f} worst={:.4f}".format(s, dt, macro, res[-1][2]))
    a = np.array(res)
    print("\nFT-Transformer: acc {:.4f}  macro-F1 {:.4f}  worst-class-F1 {:.4f}".format(
        a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()))
    print("paper cell -> {:.3f}/{:.3f}".format(a[:, 1].mean(), a[:, 2].mean()))
    print()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "unsw"
    if len(sys.argv) > 2:
        SEEDS = [int(sys.argv[2])]
    if which == "unsw":
        run("UNSW-NB15", UNSW, "attack_cat", False)
    elif which == "cic":
        run("CIC-IDS2017(dedup)", CIC, "Label", True)
