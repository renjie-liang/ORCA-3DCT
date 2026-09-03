"""Flavor B: distill a small per-token importance scorer (mask-free at inference).

Trains MLP(packed pack2x2x2 token feature [144] -> importance scalar) supervised
by the precomputed organ importance map. At inference the scorer predicts
importance from token FEATURES ALONE (no TotalSegmentator mask needed), so the
resulting `learned_score` selection is deployable without masks.

Then dumps per-volume learned importance maps to
<out>/learned/<split>/<vid>.npy (same format as organ/lesion), so it plugs into
the existing organ_mask selection machinery via --token-selection lesion_mask-
style source="learned".

Usage:
  python -m dtbd3d.importance_maps.train_token_scorer fit --train-limit 3000
  python -m dtbd3d.importance_maps.train_token_scorer dump --split valid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

REPO = Path(".")
MANIFEST = REPO / "results/reportgen_visual_tokens/e000_base_merged/manifest.json"
IMP_ROOT = Path("./data/token_importance")
SCORER_PATH = IMP_ROOT / "learned_scorer.pt"
CODEBOOK_DIM = 18
PACK = (2, 2, 2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["fit", "dump"])
    p.add_argument("--split", choices=["train", "valid"], default="valid")
    p.add_argument("--train-limit", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--label-source", choices=["organ", "lesion"], default="organ")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    return p.parse_args()


def _index(split: str) -> list[dict]:
    mani = json.loads(MANIFEST.read_text())
    idx_path = MANIFEST.parent / mani["splits"][split]["index"]
    return [json.loads(l) for l in idx_path.read_text().splitlines() if l.strip()]


def _packed_features(row: dict) -> np.ndarray:
    """Load z_quantized -> [1,D,H,W,18] -> pack2x2x2 -> [M,144]."""
    split = row["split"]
    shard = MANIFEST.parent / split / row["shard"] if (MANIFEST.parent / split / row["shard"]).exists() \
        else MANIFEST.parent / row["shard"]
    with safe_open(str(shard), framework="pt", device="cpu") as h:
        z = h.get_tensor(row["z_quantized_key"]).float().numpy()  # [18,D,H,W]
    feats = z.transpose(1, 2, 3, 0)[None]  # [1,D,H,W,18]
    ft, fh, fw = PACK
    _, td, th, tw, c = feats.shape
    pd, ph, pw = (td // ft) * ft, (th // fh) * fh, (tw // fw) * fw
    feats = feats[:, :pd, :ph, :pw, :]
    b = feats.reshape(1, pd // ft, ft, ph // fh, fh, pw // fw, fw, c)
    packed = b.transpose(0, 1, 3, 5, 2, 4, 6, 7).reshape(1, pd // ft, ph // fh, pw // fw, ft * fh * fw * c)
    return packed.reshape(-1, ft * fh * fw * c)  # [M,144]


def _labels(volume_id: str, split: str, source: str, m_grid: tuple[int, int, int]) -> np.ndarray:
    imp = np.load(IMP_ROOT / source / split / f"{volume_id}.npy").astype(np.float32)  # [16,16,8]
    x = torch.as_tensor(imp)[None, None]
    pooled = F.adaptive_avg_pool3d(x, output_size=m_grid)[0, 0].numpy()
    return pooled.reshape(-1)


class Scorer(nn.Module):
    def __init__(self, d_in: int = 144):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit(args: argparse.Namespace) -> None:
    rows = _index("train")[: args.train_limit]
    X, Y = [], []
    for i, row in enumerate(rows):
        feats = _packed_features(row)  # [M,144]
        # infer packed grid from manifest token shape (D,H,W) // pack
        d, h, w = (int(x) for x in str(row["z_quantized_shape"]).strip("[]").split(",")[1:])
        # base grid used for compression is 16x16x8 -> packed 8x8x4; but features here are at stored grid.
        # derive packed grid from feats count via cube-ish factor of M with the known pack of stored grid
        m = feats.shape[0]
        # stored grid packed dims:
        gd, gh, gw = d // PACK[0], h // PACK[1], w // PACK[2]
        if gd * gh * gw != m:
            # fallback: trust feats, approximate as near-cube
            continue
        y = _labels(row["volume_id"], "train", args.label_source, (gd, gh, gw))
        X.append(feats)
        Y.append(y)
        if (i + 1) % 500 == 0:
            print(f"loaded {i+1}/{len(rows)}", flush=True)
    X = np.concatenate(X).astype(np.float32)
    Y = np.concatenate(Y).astype(np.float32)
    print(f"train tokens: X={X.shape} Y mean={Y.mean():.3f}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xb = torch.from_numpy(X).to(dev)
    yb = torch.from_numpy(Y).to(dev)
    model = Scorer(X.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = xb.shape[0]
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for s in range(0, n, 65536):
            idx = perm[s : s + 65536]
            opt.zero_grad()
            pred = model(xb[idx])
            loss = F.mse_loss(pred, yb[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        # rank correlation proxy: corr(pred, label)
        with torch.no_grad():
            p = model(xb).cpu().numpy()
        corr = np.corrcoef(p, Y)[0, 1]
        print(f"epoch {ep}: mse={tot/n:.4f} corr(pred,label)={corr:.3f}", flush=True)
    IMP_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "d_in": X.shape[1]}, SCORER_PATH)
    print(f"saved scorer -> {SCORER_PATH}", flush=True)


def dump(args: argparse.Namespace) -> None:
    ckpt = torch.load(SCORER_PATH, map_location="cpu")
    model = Scorer(ckpt["d_in"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    out_dir = IMP_ROOT / "learned" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _index(args.split)
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_id]
    n_ok = 0
    for i, row in enumerate(rows):
        out_path = out_dir / f"{row['volume_id']}.npy"
        if out_path.exists():
            continue
        feats = _packed_features(row)
        with torch.no_grad():
            score = model(torch.from_numpy(feats.astype(np.float32))).numpy()  # [M]
        d, h, w = (int(x) for x in str(row["z_quantized_shape"]).strip("[]").split(",")[1:])
        gd, gh, gw = d // PACK[0], h // PACK[1], w // PACK[2]
        if gd * gh * gw != score.shape[0]:
            continue
        # save at base 31x32x32 grid (parity with organ/lesion maps; selection resizes down)
        grid = torch.as_tensor(score.reshape(gd, gh, gw))[None, None]
        up = F.adaptive_avg_pool3d(grid, output_size=(31, 32, 32))[0, 0].numpy().astype(np.float16)
        tmp = out_dir / f"{row['volume_id']}.tmp.npy"
        np.save(tmp, up)
        tmp.rename(out_path)
        n_ok += 1
        if (i + 1) % 500 == 0:
            print(f"dumped {i+1}/{len(rows)}", flush=True)
    print(f"DONE dump {args.split} shard {args.shard_id}: ok={n_ok}", flush=True)


def main() -> int:
    args = parse_args()
    if args.cmd == "fit":
        fit(args)
    else:
        dump(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
