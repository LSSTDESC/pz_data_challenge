#mariano.dominguez@unc.edu.ar july 2026
import argparse
import json
import pickle
import tarfile
import time
from pathlib import Path

import numpy as np
import qp
import tables_io
from rail import interactive as ri

ZMIN = 0.0
ZMAX = 3.0
NZBINS = 301
LSST_BANDS = ("u", "g", "r", "i", "z", "y")
ROMAN_BANDS = ("Y", "J", "H", "F")
MODE_TO_SELECTION = {"enf": 0, "anf": 1, "dnf": 2}
LSST_MAG_LIMITS = {
    "mag_u_lsst": 27.79,
    "mag_g_lsst": 29.04,
    "mag_r_lsst": 29.06,
    "mag_i_lsst": 28.62,
    "mag_z_lsst": 27.98,
    "mag_y_lsst": 27.05,
}
ROMAN_MAG_LIMITS = {
    "mag_Y_roman": 26.4,
    "mag_J_roman": 26.4,
    "mag_H_roman": 26.4,
    "mag_F_roman": 26.4,
}


def read_table(path):
    return tables_io.read(str(path))


def keys(table):
    if hasattr(table, "keys"):
        return set(table.keys())
    if hasattr(table, "dtype") and table.dtype.names:
        return set(table.dtype.names)
    return set()


def work(table):
    return table["photometry"] if "photometry" in keys(table) else table


def col(table, name):
    return np.asarray(work(table)[name])


def object_id(table):
    return col(table, "object_id").reshape(-1)


def build_dnf_config(table, mode="anf", use_roman=False):
    if mode not in MODE_TO_SELECTION:
        raise ValueError("Unknown mode %r" % mode)
    cols = keys(work(table))
    bands = []
    err_bands = []
    mag_limits = {}
    for band in LSST_BANDS:
        mag = "mag_%s_lsst" % band
        err = mag + "_err"
        if mag in cols and err in cols:
            bands.append(mag)
            err_bands.append(err)
            mag_limits[mag] = LSST_MAG_LIMITS[mag]
    if use_roman:
        for band in ROMAN_BANDS:
            mag = "mag_%s_roman" % band
            err = mag + "_err"
            if mag in cols and err in cols:
                bands.append(mag)
                err_bands.append(err)
                mag_limits[mag] = ROMAN_MAG_LIMITS[mag]
    if not bands:
        raise RuntimeError("No usable magnitude/error bands found")
    return {
        "hdf5_groupname": "photometry" if "photometry" in keys(table) else "",
        "bands": bands,
        "err_bands": err_bands,
        "redshift_col": "redshift",
        "mag_limits": mag_limits,
        "nondetect_val": np.nan,
        "nondetect_replace": True,
        "zmin": ZMIN,
        "zmax": ZMAX,
        "nzbins": NZBINS,
        "selection_mode": MODE_TO_SELECTION[mode],
    }


def unwrap(obj):
    if hasattr(obj, "data") and getattr(obj, "data") is not None:
        return getattr(obj, "data")
    for meth in ("get_data", "read"):
        if hasattr(obj, meth):
            try:
                val = getattr(obj, meth)()
                if val is not None:
                    return val
            except Exception:
                pass
    return obj


def train_raw_dnf_table(training_table, mode="anf", use_roman=False):
    cfg = build_dnf_config(training_table, mode=mode, use_roman=use_roman)
    result = ri.estimation.algos.dnf.dnf_informer(training_data=training_table, **cfg)
    return unwrap(result["model"])


def estimate_raw_dnf_table(model, input_table, mode="anf", use_roman=False):
    cfg = build_dnf_config(input_table, mode=mode, use_roman=use_roman)
    result = ri.estimation.algos.dnf.dnf_estimator(input_data=input_table, model=model, **cfg)
    return unwrap(result["output"])


def gaussian_pdf_grid(zgrid, mu, sigma):
    mu = np.asarray(mu, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1.0e-6), sigma, 1.0e-6)
    y = np.exp(-0.5 * ((zgrid[None, :] - mu[:, None]) / sigma[:, None]) ** 2)
    norm = np.trapezoid(y, zgrid, axis=1)
    bad = ~np.isfinite(norm) | (norm <= 0)
    if np.any(bad):
        y[bad, :] = 0.0
        idx = np.searchsorted(zgrid, np.clip(mu[bad], zgrid[0], zgrid[-1]))
        idx = np.clip(idx, 0, len(zgrid) - 1)
        y[np.where(bad)[0], idx] = 1.0
        norm = np.trapezoid(y, zgrid, axis=1)
    return y / norm[:, None]


def fallback_sigma_from_ancil(ancil, sigma_min=1.0e-4):
    required = ["photozerr_param", "photozerr_fit", "photozerr"]
    missing = [k for k in required if k not in ancil]
    if missing:
        raise RuntimeError("DNF output missing required ancillary fields: %s" % missing)
    param = np.asarray(ancil["photozerr_param"], dtype=float).reshape(-1)
    fit = np.asarray(ancil["photozerr_fit"], dtype=float).reshape(-1)
    fallback = np.asarray(ancil["photozerr"], dtype=float).reshape(-1)
    base = np.sqrt(param**2 + fit**2)
    use_fallback = (param == 0.0) & (fit == 0.0)
    sigma = np.where(use_fallback, fallback, base)
    sigma = np.where(np.isfinite(sigma), sigma, sigma_min)
    sigma = np.maximum(sigma, sigma_min)
    return sigma


def rebuild_fallback_ensemble(raw_ensemble, input_table, zmode_policy="center_clipped"):
    ancil = dict(raw_ensemble.ancil or {})
    if "DNF_Z" not in ancil:
        raise RuntimeError("DNF output missing required ancillary field: DNF_Z")
    center = np.asarray(ancil["DNF_Z"], dtype=float).reshape(-1)
    sigma = fallback_sigma_from_ancil(ancil)
    if zmode_policy == "center_clipped":
        zmode = np.clip(center, ZMIN, ZMAX)
    elif zmode_policy == "keep_existing" and "zmode" in ancil:
        zmode = np.asarray(ancil["zmode"], dtype=float).reshape(-1)
    else:
        raise ValueError("zmode_policy must be center_clipped or keep_existing")
    zgrid = np.linspace(ZMIN, ZMAX, NZBINS)
    yvals = gaussian_pdf_grid(zgrid, center, sigma)
    new_ancil = dict(ancil)
    if "object_id" not in new_ancil:
        new_ancil["object_id"] = object_id(input_table)
    new_ancil["zmode"] = zmode
    new_ancil["photozerr_original_dnf"] = np.asarray(ancil["photozerr"], dtype=float).reshape(-1)
    new_ancil["photozerr"] = sigma
    new_ancil["pdf_sigma_fallback"] = sigma
    new_ancil["pdf_calib_beta_neig"] = np.zeros(len(center))
    new_ancil["pdf_calib_sigma_floor"] = np.zeros(len(center))
    return qp.Ensemble(qp.interp, data={"xvals": zgrid, "yvals": yvals}, ancil=new_ancil)


def train_model(train_file, model_file, mode="anf", use_roman=False):
    training_table = read_table(train_file)
    raw_model = train_raw_dnf_table(training_table, mode=mode, use_roman=use_roman)
    wrapped = {
        "raw_model": raw_model,
        "method": "dnf_fallback_pdf",
        "mode": mode,
        "use_roman": bool(use_roman),
        "pdf_rule": "photozerr fallback if param=fit=0 else sqrt(param^2+fit^2)",
        "zmin": ZMIN,
        "zmax": ZMAX,
        "nzbins": NZBINS,
    }
    model_file = Path(model_file)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    with open(model_file, "wb") as fout:
        pickle.dump(wrapped, fout, protocol=pickle.HIGHEST_PROTOCOL)
    return wrapped


def load_model(model_file):
    with open(model_file, "rb") as fin:
        model = pickle.load(fin)
    if isinstance(model, dict) and "raw_model" in model:
        return model
    return {"raw_model": model, "method": "dnf_raw_legacy", "mode": "anf", "use_roman": False}


def estimate_with_model(model_file, test_file, output_file, zmode_policy="center_clipped"):
    wrapped = load_model(model_file)
    input_table = read_table(test_file)
    raw_ensemble = estimate_raw_dnf_table(
        wrapped["raw_model"],
        input_table,
        mode=wrapped.get("mode", "anf"),
        use_roman=bool(wrapped.get("use_roman", False)),
    )
    final_ensemble = rebuild_fallback_ensemble(raw_ensemble, input_table, zmode_policy=zmode_policy)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_ensemble.write_to(str(output_file))


def train_and_estimate(train_file, test_file, output_file, mode="anf", use_roman=False, zmode_policy="center_clipped"):
    training_table = read_table(train_file)
    input_table = read_table(test_file)
    raw_model = train_raw_dnf_table(training_table, mode=mode, use_roman=use_roman)
    raw_ensemble = estimate_raw_dnf_table(raw_model, input_table, mode=mode, use_roman=use_roman)
    final_ensemble = rebuild_fallback_ensemble(raw_ensemble, input_table, zmode_policy=zmode_policy)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_ensemble.write_to(str(output_file))


def prefix(taskset, sim):
    return "pz_challenge_taskset_%d_%s" % (taskset, sim)


def make_tarball(run_dir, output_tar=None):
    run_dir = Path(run_dir)
    output_tar = Path(output_tar) if output_tar else run_dir / (run_dir.name + "_submission.tgz")
    files = []
    files.extend(sorted((run_dir / "test_outputs").glob("pz_challenge_taskset_*_pz_estimate_*.hdf5")))
    files.extend(sorted((run_dir / "models").glob("pz_challenge_taskset_*_pz_model_*.pkl")))
    if not files:
        raise RuntimeError("No submission files found in %s" % run_dir)
    with tarfile.open(output_tar, "w:gz") as tar:
        for file_path in files:
            tar.add(file_path, arcname=file_path.name)
    return output_tar


def run_one_case(public_area, run_dir, taskset, sim, scenario, mode="anf", use_roman=False, kind="test"):
    public_area = Path(public_area)
    run_dir = Path(run_dir)
    pfx = prefix(taskset, sim)
    train_file = public_area / (pfx + "_training_" + scenario + ".hdf5")
    test_file = public_area / (pfx + "_test_" + scenario + ".hdf5")
    model_file = run_dir / "models" / (pfx + "_pz_model_" + scenario + ".pkl")
    test_output = run_dir / "test_outputs" / (pfx + "_pz_estimate_" + scenario + ".hdf5")
    traintrain_output = run_dir / "traintrain_outputs" / (pfx + "_pz_evaluation_" + scenario + ".hdf5")
    timing = {}
    t0 = time.perf_counter()
    train_model(train_file, model_file, mode=mode, use_roman=use_roman)
    timing["train_seconds"] = time.perf_counter() - t0
    if kind in ("test", "both"):
        t0 = time.perf_counter()
        estimate_with_model(model_file, test_file, test_output)
        timing["test_estimate_seconds"] = time.perf_counter() - t0
    if kind in ("traintrain", "both"):
        t0 = time.perf_counter()
        estimate_with_model(model_file, train_file, traintrain_output)
        timing["traintrain_estimate_seconds"] = time.perf_counter() - t0
    timing_file = run_dir / "timing" / (pfx + "_" + scenario + ".json")
    timing_file.parent.mkdir(parents=True, exist_ok=True)
    timing_file.write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n")
    return timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-area", default="public")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--taskset", type=int, choices=[1, 2], required=True)
    parser.add_argument("--sim", choices=["cardinal", "flagship"], required=True)
    parser.add_argument("--scenario", choices=["1yr", "10yr"], required=True)
    parser.add_argument("--mode", choices=["enf", "anf", "dnf"], default="anf")
    parser.add_argument("--use-roman", action="store_true")
    parser.add_argument("--kind", choices=["traintrain", "test", "both"], default="test")
    parser.add_argument("--make-tar", action="store_true")
    args = parser.parse_args()
    timing = run_one_case(args.public_area, args.run_dir, args.taskset, args.sim, args.scenario, mode=args.mode, use_roman=args.use_roman, kind=args.kind)
    print(json.dumps(timing, indent=2, sort_keys=True))
    if args.make_tar:
        print("Wrote", make_tarball(args.run_dir))


if __name__ == "__main__":
    main()
