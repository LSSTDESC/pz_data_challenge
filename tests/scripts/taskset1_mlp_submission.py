#!/usr/bin/env python3
"""Taskset 1 submission helpers for Simple MLP Flow Matching.

This script turns a trained Simple MLP Flow model into official challenge-style
`qp` outputs and exposes the required estimation-only function:

    run_taskset_1_estimation_only(model_file, test_file, output_file)

The output file is written in `qp` HDF5 format and includes ancillary
`object_id` and `zmode` arrays, matching the challenge validation checks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from train_simple_mlp_flow_torch import FlowMLP, read_table, sample_posterior


def _require_qp():
    try:
        import qp  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on external env
        raise SystemExit(
            "This script requires the `qp` package. Install it in the "
            "submission environment before running."
        ) from exc
    return qp


def _load_model_bundle(model_file: str | Path, device: torch.device) -> tuple[FlowMLP, dict[str, Any]]:
    bundle = torch.load(model_file, map_location=device)
    metadata = dict(bundle["metadata"])
    feature_cols = list(metadata["feature_cols"])
    hidden = int(metadata["hidden"])
    depth = int(metadata["depth"])
    dropout = float(metadata.get("dropout", 0.0))
    model = FlowMLP(len(feature_cols) + 2, hidden, depth, dropout).to(device)
    model.load_state_dict(bundle["model_state"])
    model.eval()
    return model, metadata


def _prepare_test_features(df: pd.DataFrame, metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    feature_cols = list(metadata["feature_cols"])
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise SystemExit(f"Test file is missing required feature columns: {missing}")
    if "object_id" not in df.columns:
        raise SystemExit("Test file must include an `object_id` column.")

    features_raw = df[feature_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    x_mean = np.asarray(metadata["x_mean"], dtype=np.float32)
    x_std = np.asarray(metadata["x_std"], dtype=np.float32)
    x_std = np.where(x_std == 0, 1.0, x_std)

    # Keep every test object so object_id order matches the challenge file.
    # We fill missing feature values with the training-set feature mean.
    fill_values = x_mean[None, :]
    features_raw = np.where(np.isfinite(features_raw), features_raw, fill_values)
    features = (features_raw - x_mean[None, :]) / x_std[None, :]
    object_ids = df["object_id"].to_numpy()
    return features.astype(np.float32), object_ids


def _samples_to_interp_grid(samples: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    """Convert posterior samples into smooth densities on a shared z-grid."""
    n_obj, n_samples = samples.shape
    densities = np.empty((n_obj, z_grid.size), dtype=np.float32)

    for i in range(n_obj):
        s = np.asarray(samples[i], dtype=np.float64)
        s = s[np.isfinite(s)]
        if s.size == 0:
            densities[i] = np.full(z_grid.shape, 1.0 / (z_grid[-1] - z_grid[0]), dtype=np.float32)
            continue

        std = float(np.std(s))
        iqr = float(np.subtract(*np.percentile(s, [75, 25])))
        sigma = min(std, iqr / 1.34) if iqr > 0 else std
        if not np.isfinite(sigma) or sigma <= 1e-4:
            sigma = max((z_grid[-1] - z_grid[0]) / 200.0, 1e-3)
        bandwidth = 0.9 * sigma * (max(s.size, 2) ** (-1.0 / 5.0))
        bandwidth = max(float(bandwidth), 1e-3)

        diff = (z_grid[:, None] - s[None, :]) / bandwidth
        kernel = np.exp(-0.5 * diff * diff) / (np.sqrt(2.0 * np.pi) * bandwidth)
        pdf = np.mean(kernel, axis=1)
        area = np.trapezoid(pdf, z_grid)
        if not np.isfinite(area) or area <= 0:
            pdf = np.full(z_grid.shape, 1.0 / (z_grid[-1] - z_grid[0]), dtype=np.float64)
        else:
            pdf = pdf / area
        densities[i] = pdf.astype(np.float32)
    return densities


def _attach_ancillary(ensemble, object_ids: np.ndarray, zmode: np.ndarray) -> None:
    ancil = {
        "object_id": np.asarray(object_ids),
        "zmode": np.asarray(zmode, dtype=np.float32),
    }

    # `qp` versions differ a bit in how ancillary data are attached.
    if hasattr(ensemble, "set_ancil"):
        ensemble.set_ancil(ancil)
        return
    if hasattr(ensemble, "set_ancillary"):
        ensemble.set_ancillary(ancil)
        return
    if hasattr(ensemble, "set_ancillary_data"):
        ensemble.set_ancillary_data(ancil)
        return
    if hasattr(ensemble, "ancil"):
        ensemble.ancil = ancil
        return
    if hasattr(ensemble, "ancillary"):
        ensemble.ancillary = ancil
        return
    if hasattr(ensemble, "ancillary_data"):
        ensemble.ancillary_data = ancil
        return
    raise AttributeError("Could not find a supported way to attach ancillary data to this qp.Ensemble.")


def export_model_predictions_to_qp(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
    *,
    posterior_samples: int = 128,
    sample_steps: int = 64,
    z_grid_size: int = 301,
    device: str | None = None,
) -> None:
    qp = _require_qp()

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, metadata = _load_model_bundle(model_file, resolved_device)
    test_df = read_table(Path(test_file))
    features, object_ids = _prepare_test_features(test_df, metadata)
    x_tensor = torch.tensor(features, dtype=torch.float32)

    samples = sample_posterior(
        model,
        x_tensor,
        posterior_samples,
        sample_steps,
        float(metadata["z_min"]),
        float(metadata["z_max"]),
        resolved_device,
    )

    z_grid = np.linspace(float(metadata["z_min"]), float(metadata["z_max"]), z_grid_size, dtype=np.float32)
    densities = _samples_to_interp_grid(samples, z_grid)
    ensemble = qp.interp.create_ensemble(z_grid, densities)
    zmode = z_grid[np.argmax(densities, axis=1)]
    _attach_ancillary(ensemble, object_ids, zmode)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble.write_to(output_path)


def run_taskset_1_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """Challenge subtask 2 entry point for taskset 1."""
    export_model_predictions_to_qp(model_file, test_file, output_file)

def run_taskset_2_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """Challenge subtask 2 entry point for taskset 2."""
    export_model_predictions_to_qp(model_file, test_file, output_file)

def main() -> None:
    parser = argparse.ArgumentParser(description="Export Simple MLP Flow predictions to official qp format.")
    parser.add_argument("--model-file", required=True, help="Path to simple_mlp_flow_model.pt")
    parser.add_argument("--test-file", required=True, help="Path to taskset test file (.hdf5/.pq/.csv)")
    parser.add_argument("--output-file", required=True, help="Target qp HDF5 file")
    parser.add_argument("--posterior-samples", type=int, default=128)
    parser.add_argument("--sample-steps", type=int, default=64)
    parser.add_argument("--z-grid-size", type=int, default=301)
    parser.add_argument("--device", default=None, help="Override torch device, e.g. cpu or cuda")
    args = parser.parse_args()

    export_model_predictions_to_qp(
        model_file=args.model_file,
        test_file=args.test_file,
        output_file=args.output_file,
        posterior_samples=args.posterior_samples,
        sample_steps=args.sample_steps,
        z_grid_size=args.z_grid_size,
        device=args.device,
    )
    print(f"wrote {args.output_file}", flush=True)


if __name__ == "__main__":
    main()
