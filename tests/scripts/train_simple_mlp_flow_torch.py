#!/usr/bin/env python3
"""Train Simple MLP conditional rectified flow with PyTorch/GPU.

Output is compatible with rail_style_eval.py:
  simple_mlp_flow_predictions.npz with z_true, samples, and optional mag_i/mag_r.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


MAG_CANDIDATES = ["mag_i", "mag_i_lsst", "i", "i_mag", "mag_r", "mag_r_lsst", "r", "r_mag"]
TARGET_CANDIDATES = ["redshift", "z", "z_true", "zref", "z_ref", "true_redshift"]


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".npz":
        data = np.load(path)
        return pd.DataFrame({k: data[k] for k in data.files if data[k].ndim == 1})
    if suffix in {".pq", ".parquet"}:
        return pd.read_parquet(path)
    if suffix in {".hdf5", ".h5"}:
        import h5py

        arrays = {}
        with h5py.File(path, "r") as f:
            for key in f.keys():
                obj = f[key]
                if hasattr(obj, "shape") and len(obj.shape) == 1:
                    arrays[key] = obj[()]
        return pd.DataFrame(arrays)
    raise SystemExit("Input must be CSV, NPZ, parquet, or HDF5.")


def choose_column(columns, explicit, candidates, kind):
    if explicit:
        if explicit not in columns:
            raise SystemExit(f"{kind} column '{explicit}' not found. Available columns: {list(columns)}")
        return explicit
    normalized = {c.lower().replace("-", "_"): c for c in columns}
    for cand in candidates:
        if cand in normalized:
            return normalized[cand]
    raise SystemExit(f"Could not infer {kind} column. Use --{kind}-col. Available columns: {list(columns)}")


def choose_features(df: pd.DataFrame, target_col: str, feature_cols: str | None) -> list[str]:
    if feature_cols:
        cols = [c.strip() for c in feature_cols.split(",") if c.strip()]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(f"Feature columns not found: {missing}")
        return cols
    cols = []
    for c in df.columns:
        lc = c.lower()
        if c == target_col:
            continue
        if any(token in lc for token in ["mag", "flux", "err", "snr", "color"]):
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
    if not cols:
        raise SystemExit("Could not infer feature columns. Use --feature-cols.")
    return cols


class FlowMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, depth: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([features, z_t[:, None], t[:, None]], dim=1)
        return self.net(x).squeeze(1)


def split_indices(n: int, val_frac: float, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test = idx[:n_test]
    val = idx[n_test : n_test + n_val]
    train = idx[n_test + n_val :]
    return train, val, test


def load_pretrained_weights(model: nn.Module, checkpoint_path: Path) -> dict:
    bundle = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(bundle, dict) or "model_state" not in bundle:
        raise SystemExit(f"Invalid checkpoint format in {checkpoint_path}")
    state = bundle["model_state"]
    current = model.state_dict()
    bad_shapes = []
    for key, tensor in state.items():
        if key in current and current[key].shape != tensor.shape:
            bad_shapes.append((key, tuple(tensor.shape), tuple(current[key].shape)))
    if bad_shapes:
        lines = [
            f"{key}: checkpoint {old_shape} vs current {new_shape}"
            for key, old_shape, new_shape in bad_shapes
        ]
        raise SystemExit(
            "Pretrained checkpoint is not architecture-compatible.\n"
            + "\n".join(lines)
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            "Loaded pretrained checkpoint with partial match:"
            f" missing={missing}, unexpected={unexpected}",
            flush=True,
        )
    return bundle.get("metadata", {})


def flow_loss(model, features, z_true):
    z0 = torch.randn_like(z_true)
    t = torch.rand_like(z_true)
    zt = (1.0 - t) * z0 + t * z_true
    target_v = z_true - z0
    pred_v = model(features, zt, t)
    return torch.mean((pred_v - target_v) ** 2)


@torch.no_grad()
def sample_posterior(model, features, n_samples, n_steps, z_min, z_max, device, batch_objects=4096):
    model.eval()
    all_samples = []
    dt = 1.0 / n_steps
    for start in range(0, len(features), batch_objects):
        x = features[start : start + batch_objects].to(device)
        m = x.shape[0]
        z = torch.randn(m, n_samples, device=device)
        flat_x = x[:, None, :].expand(m, n_samples, x.shape[1]).reshape(m * n_samples, x.shape[1])
        z_flat = z.reshape(-1)
        for step in range(n_steps):
            t_value = torch.full_like(z_flat, (step + 0.5) * dt)
            v = model(flat_x, z_flat, t_value)
            z_flat = z_flat + dt * v
        z = z_flat.reshape(m, n_samples).clamp(z_min, z_max)
        all_samples.append(z.cpu().numpy())
    return np.concatenate(all_samples, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--feature-cols", default=None)
    parser.add_argument("--outdir", default="simple_mlp_flow_torch_outputs")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--posterior-samples", type=int, default=128)
    parser.add_argument("--sample-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--init-model", default=None, help="Optional path to pretrained simple_mlp_flow_model.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.split_seed is None:
        args.split_seed = args.seed
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    df = read_table(Path(args.input))
    if args.max_rows is not None and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)

    target_col = choose_column(df.columns, args.target_col, TARGET_CANDIDATES, "target")
    feature_cols = choose_features(df, target_col, args.feature_cols)
    mag_col = next((c for c in MAG_CANDIDATES if c in df.columns), None)

    use_cols = feature_cols + [target_col] + ([mag_col] if mag_col and mag_col not in feature_cols else [])
    clean = df[use_cols].replace([np.inf, -np.inf], np.nan).dropna()
    features_raw = clean[feature_cols].to_numpy(dtype=np.float32)
    z = clean[target_col].to_numpy(dtype=np.float32)

    x_mean = features_raw.mean(axis=0)
    x_std = features_raw.std(axis=0)
    x_std[x_std == 0] = 1.0
    x = (features_raw - x_mean) / x_std
    z_min = max(0.0, float(np.min(z)) - 0.05)
    z_max = float(np.max(z)) + 0.05

    train_idx, val_idx, test_idx = split_indices(len(z), args.val_frac, args.test_frac, args.split_seed)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    z_tensor = torch.tensor(z, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(x_tensor[train_idx], z_tensor[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_x = x_tensor[val_idx].to(device)
    val_z = z_tensor[val_idx].to(device)

    model = FlowMLP(x.shape[1] + 2, args.hidden, args.depth, args.dropout).to(device)
    init_metadata = None
    if args.init_model:
        init_metadata = load_pretrained_weights(model, Path(args.init_model))
        print(f"Initialized model from pretrained checkpoint: {args.init_model}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_rows = ["epoch,train_loss,val_loss"]
    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, zb in train_loader:
            xb = xb.to(device)
            zb = zb.to(device)
            loss = flow_loss(model, xb, zb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            val_loss = float(flow_loss(model, val_x, val_z).detach().cpu())
        train_loss = float(np.mean(losses))
        log_rows.append(f"{epoch},{train_loss},{val_loss}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % max(1, args.epochs // 20) == 0:
            print(f"epoch {epoch:4d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    metadata = {
        "input": args.input,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "mag_col": mag_col,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "z_min": z_min,
        "z_max": z_max,
        "hidden": args.hidden,
        "depth": args.depth,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "device": str(device),
        "best_val_loss": best_val,
        "init_model": args.init_model,
        "init_metadata": init_metadata,
    }
    torch.save({"model_state": model.state_dict(), "metadata": metadata}, outdir / "simple_mlp_flow_model.pt")
    (outdir / "training_log.csv").write_text("\n".join(log_rows) + "\n", encoding="utf-8")
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    samples = sample_posterior(
        model,
        x_tensor[test_idx],
        args.posterior_samples,
        args.sample_steps,
        z_min,
        z_max,
        device,
    )
    output = {
        "z_true": z[test_idx],
        "samples": samples,
    }
    if mag_col:
        mag_values = clean[mag_col].to_numpy(dtype=np.float32)[test_idx]
        key = "mag_i" if "i" in mag_col.lower() else "mag_r"
        output[key] = mag_values
    np.savez(outdir / "simple_mlp_flow_predictions.npz", **output)
    print(f"wrote {outdir / 'simple_mlp_flow_predictions.npz'}", flush=True)


if __name__ == "__main__":
    main()
