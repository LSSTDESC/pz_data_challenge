#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C-4 inference: C-3 distance-adaptive PDF + local photometric SPEC selector.

This version keeps the C-4 decision rule unchanged, but adds:
  * find_best_bmu re-export from C-3 for a single public inference module;
  * save_qp_submission, so benchmark/inference scripts do not need to import
    a second module;
  * optional CuPy acceleration for BMU assignment while building the C-4
    local lookup. If CuPy/GPU is unavailable, it falls back to CPU.

The C-4 selector itself is kept numerically equivalent to the original:
    f_highz >= 0.50 and delta_local >= 0.05
"""

import pickle
import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except Exception:
    cp = None
    _CUPY_AVAILABLE = False



# ============================================================
# MODEL LOADING
# ============================================================

def load_submission_model(model_file):

    with open(
        model_file,
        "rb"
    ) as f:

        model = pickle.load(f)


    required_keys = [
        "regions",
        "metadata"
    ]


    for key in required_keys:

        if key not in model:

            raise ValueError(
                f"Invalid Co-SOM model. Missing key: {key}"
            )


    if "preprocess_params" not in model["metadata"]:

        raise ValueError(
            "Missing preprocessing parameters"
        )


    if "z_edges_global" not in model["metadata"]:

        raise ValueError(
            "Missing redshift grid"
        )


    return model

# ============================================================
# BMU SEARCH
# ============================================================

def find_best_bmu(X, models, k_candidates=5):
    """
    Find the best matching units (BMUs) for each galaxy among all trained
    Co-SOM regions and SOM maps.

    The k closest BMU candidates are stored for each object. This allows
    fallback to neighbouring SOM neurons when the closest neuron does not
    contain a valid redshift PDF.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.

    models : list
        Trained Co-SOM models.

    k_candidates : int
        Number of BMU candidates stored per object.

    Returns
    -------
    matches : dict
        Best BMU information plus candidate BMUs.
    """

    N = X.shape[0]


    # ==========================================================
    # Best global match
    # ==========================================================

    best_distance = np.full(
        N,
        np.inf,
        dtype=np.float32
    )


    best_region = np.zeros(
        N,
        dtype=np.int32
    )


    best_som = np.zeros(
        N,
        dtype=np.int32
    )


    best_bmu = np.zeros(
        N,
        dtype=np.int32
    )


    # ==========================================================
    # Store top-k candidates
    # ==========================================================

    candidate_distance = np.full(
        (N, k_candidates),
        np.inf,
        dtype=np.float32
    )


    candidate_region = np.zeros(
        (N, k_candidates),
        dtype=np.int32
    )


    candidate_som = np.zeros(
        (N, k_candidates),
        dtype=np.int32
    )


    candidate_bmu = np.zeros(
        (N, k_candidates),
        dtype=np.int32
    )


    # ==========================================================
    # GPU / CPU fallback
    # ==========================================================
    use_gpu = gpu_available()

    if use_gpu:
        X_gpu = cp.asarray(
            X,
            dtype=cp.float32
        )
    else:
        X_cpu = np.asarray(
            X,
            dtype=np.float32
        )

    # ==========================================================
    # Search all SOMs
    # ==========================================================

    for region_id, model in enumerate(models):

        for som_id in [0, 1]:

            som = model[
                f"som_{som_id}"
            ]

            if use_gpu:

                weights_gpu = cp.asarray(
                    som["weights"],
                    dtype=cp.float32
                )

                w2 = cp.sum(
                    weights_gpu * weights_gpu,
                    axis=1
                )

                x2 = cp.sum(
                    X_gpu * X_gpu,
                    axis=1,
                    keepdims=True
                )

                dists = cp.maximum(
                    x2
                    + w2
                    - 2 * X_gpu @ weights_gpu.T,
                    0.0
                )

                bmu = cp.argmin(
                    dists,
                    axis=1
                )

                distance = cp.sqrt(
                    cp.min(
                        dists,
                        axis=1
                    )
                )

                bmu_cpu = cp.asnumpy(
                    bmu
                )

                distance_cpu = cp.asnumpy(
                    distance
                )

                del weights_gpu
                del dists

            else:

                weights_cpu = np.asarray(
                    som["weights"],
                    dtype=np.float32
                )

                bmu_cpu = _cpu_bmu(
                    X_cpu,
                    weights_cpu
                )

                selected_weights = weights_cpu[bmu_cpu]

                delta = (
                    X_cpu
                    - selected_weights
                )

                distance_cpu = np.sqrt(
                    np.sum(
                        delta * delta,
                        axis=1
                    )
                ).astype(np.float32)

            # ==================================================
            # Update global top-k candidates
            # ==================================================

            new_distance = np.column_stack(
                [
                    candidate_distance,
                    distance_cpu
                ]
            )


            new_region = np.column_stack(
                [
                    candidate_region,
                    np.full(
                        N,
                        region_id
                    )
                ]
            )


            new_som = np.column_stack(
                [
                    candidate_som,
                    np.full(
                        N,
                        som_id
                    )
                ]
            )


            new_bmu = np.column_stack(
                [
                    candidate_bmu,
                    bmu_cpu
                ]
            )


            idx = np.argsort(
                new_distance,
                axis=1
            )[:, :k_candidates]


            rows = np.arange(N)[:,None]


            candidate_distance = (
                new_distance[
                    rows,
                    idx
                ]
            )

            candidate_region = (
                new_region[
                    rows,
                    idx
                ]
            )

            candidate_som = (
                new_som[
                    rows,
                    idx
                ]
            )

            candidate_bmu = (
                new_bmu[
                    rows,
                    idx
                ]
            )



    # ==========================================================
    # Best match = first candidate
    # ==========================================================

    best_distance = candidate_distance[:,0]

    best_region = candidate_region[:,0]

    best_som = candidate_som[:,0]

    best_bmu = candidate_bmu[:,0]


    return {

        "region": best_region.astype(np.int32),

        "som_id": best_som.astype(np.int32),

        "bmu": best_bmu.astype(np.int32),

        "distance": best_distance,


        # New information
        "bmu_candidates":
            candidate_bmu.astype(np.int32),

        "region_candidates":
            candidate_region.astype(np.int32),

        "som_candidates":
            candidate_som.astype(np.int32),

        "distance_candidates":
            candidate_distance,

        # C-4 local selector uses the original test feature vector.
        "X_test": np.asarray(X, dtype=np.float32)

    }

# ============================================================
# C-3 BASELINE
# ============================================================

def _valid_pdf(pdf):
    return pdf is not None and np.all(np.isfinite(pdf)) and np.sum(pdf) > 0

def _candidate_count(distance):
    if distance <= 0.5:
        return 1
    if distance < 1.0:
        return 3
    if distance < 1.5:
        return 5
    return 10

def predict_objects_C3(matches, regions, z_edges_global=None, sigma_distance=0.5):
    n_objects = len(matches["bmu"])
    n_candidates = matches["bmu_candidates"].shape[1]
    z_pred = np.full(n_objects, np.nan, dtype=np.float32)
    pdf_list = []
    used_region = np.zeros(n_objects, dtype=np.int32)
    used_som = np.zeros(n_objects, dtype=np.int32)
    used_bmu = np.zeros(n_objects, dtype=np.int32)
    mixture_counter = 0
    fallback_counter = 0

    for i in range(n_objects):
        d0 = float(matches["distance"][i])
        target_n = min(_candidate_count(d0), n_candidates)
        valid_pdfs, valid_distances, valid_meta = [], [], []

        for j in range(target_n):
            region = int(matches["region_candidates"][i, j])
            som_id = int(matches["som_candidates"][i, j])
            bmu = int(matches["bmu_candidates"][i, j])
            distance = float(matches["distance_candidates"][i, j])
            if not np.isfinite(distance):
                continue
            som = regions[region][f"som_{som_id}"]
            p = np.asarray(som["pdf"][bmu], dtype=np.float32)
            if not _valid_pdf(p):
                continue
            valid_pdfs.append(p)
            valid_distances.append(distance)
            valid_meta.append((region, som_id, bmu))

        if not valid_pdfs:
            region = int(matches["region"][i]); som_id = int(matches["som_id"][i]); bmu = int(matches["bmu"][i])
            som = regions[region][f"som_{som_id}"]
            n_bins = som["pdf"].shape[1]
            final_pdf = np.ones(n_bins, dtype=np.float32) / n_bins
            used_region[i], used_som[i], used_bmu[i] = region, som_id, bmu
            pdf_list.append(final_pdf)
            fallback_counter += 1
            continue

        normalized = [p.astype(np.float64) / p.sum() for p in valid_pdfs]

        if len(normalized) == 1 or d0 <= 0.5:
            final_pdf = normalized[0]
            selected_meta = valid_meta[0]
        else:
            distances = np.asarray(valid_distances, dtype=np.float64)
            delta = np.maximum(distances - distances[0], 0.0)
            weights = np.exp(-0.5 * (delta / max(float(sigma_distance), 1e-8)) ** 2)
            weights /= weights.sum()
            final_pdf = np.zeros_like(normalized[0])
            for w, p in zip(weights, normalized):
                final_pdf += w * p
            final_pdf /= final_pdf.sum()
            selected_meta = valid_meta[0]
            mixture_counter += 1

        region, som_id, bmu = selected_meta
        used_region[i], used_som[i], used_bmu[i] = region, som_id, bmu
        som = regions[region][f"som_{som_id}"]

        if d0 <= 0.5 or len(normalized) == 1:
            z_pred[i] = np.float32(som["z_median"][bmu])
        else:
            # Use the stored global redshift grid when available.
            z_edges = z_edges_global
            if z_edges is not None and len(z_edges) == len(final_pdf) + 1:
                cdf = np.cumsum(final_pdf); cdf /= cdf[-1]
                z_centers = 0.5 * (np.asarray(z_edges[:-1]) + np.asarray(z_edges[1:]))
                idx = int(np.searchsorted(cdf, 0.5, side="left"))
                idx = min(max(idx, 0), len(z_centers) - 1)
                z_pred[i] = np.float32(z_centers[idx])
            else:
                # The model normally carries the global grid in metadata;
                # if not present here, use the weighted mean bin index only
                # as a safe fallback to the nearest-BMU point estimate.
                z_pred[i] = np.float32(som["z_median"][bmu])

        pdf_list.append(final_pdf.astype(np.float32))

    print(f"C-3 PDF mixtures used: {mixture_counter} / {n_objects}")
    print(f"Fallback/non-finite candidate sets: {fallback_counter} / {n_objects}")
    return {
        "region": used_region,
        "som_id": used_som,
        "bmu": used_bmu,
        "bmu_original": matches["bmu"],
        "distance": matches["distance"],
        "z": z_pred,
        "pdf": np.asarray(pdf_list, dtype=np.float32),
    }

# ============================================================
# C-4 INFERENCE
# ============================================================

def gpu_available():
    """Return True when CuPy can actually access a CUDA device."""
    if not _CUPY_AVAILABLE:
        return False
    try:
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def _cpu_bmu(X, weights, chunk=4096):
    X = np.asarray(X, dtype=np.float32)
    W = np.asarray(weights, dtype=np.float32).reshape(-1, X.shape[1])
    out = np.empty(len(X), dtype=np.int32)
    w2 = np.sum(W * W, axis=1)
    for start in range(0, len(X), chunk):
        stop = min(start + chunk, len(X))
        xx = X[start:stop]
        d2 = np.sum(xx * xx, axis=1)[:, None] + w2[None, :] - 2.0 * (xx @ W.T)
        out[start:stop] = np.argmin(np.maximum(d2, 0.0), axis=1)
    return out


def _gpu_bmu(X, weights, chunk=16384):
    """GPU BMU assignment using CuPy, chunked to limit VRAM use."""
    X = np.asarray(X, dtype=np.float32)
    W = np.asarray(weights, dtype=np.float32).reshape(-1, X.shape[1])
    out = np.empty(len(X), dtype=np.int32)
    xg = cp.asarray(X)
    wg = cp.asarray(W)
    w2 = cp.sum(wg * wg, axis=1)
    for start in range(0, len(X), chunk):
        stop = min(start + chunk, len(X))
        xx = xg[start:stop]
        d2 = cp.sum(xx * xx, axis=1)[:, None] + w2[None, :] - 2.0 * (xx @ wg.T)
        out[start:stop] = cp.asnumpy(cp.argmin(cp.maximum(d2, 0.0), axis=1))
    return out


def _bmu(X, weights, use_gpu=True):
    if use_gpu and gpu_available():
        return _gpu_bmu(X, weights)
    return _cpu_bmu(X, weights)


def _nearest_indices(query, X, k=20, chunk=4096):
    query = np.asarray(query, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)
    if len(X) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    k = min(int(k), len(X))
    best_d2 = np.full(k, np.inf, dtype=np.float64)
    best_i = np.full(k, -1, dtype=np.int64)
    q2 = float(np.dot(query, query))
    for start in range(0, len(X), chunk):
        stop = min(start + chunk, len(X))
        xx = X[start:stop]
        d2 = q2 + np.sum(xx * xx, axis=1) - 2.0 * (xx @ query)
        d2 = np.maximum(d2.astype(np.float64), 0.0)
        take = min(k, len(d2))
        idx_local = np.argpartition(d2, take - 1)[:take]
        cand_d = d2[idx_local]
        cand_i = idx_local.astype(np.int64) + start
        merged_d = np.concatenate([best_d2, cand_d])
        merged_i = np.concatenate([best_i, cand_i])
        keep = np.argpartition(merged_d, k - 1)[:k]
        best_d2 = merged_d[keep]
        best_i = merged_i[keep]
    order = np.argsort(best_d2)
    return best_i[order], np.sqrt(np.maximum(best_d2[order], 0.0))


def _gaussian_weights(distances):
    d = np.asarray(distances, dtype=np.float64)
    if len(d) == 0:
        return d
    d0 = float(np.min(d))
    scale = max(float(np.median(d)), 1e-8)
    w = np.exp(-0.5 * ((d - d0) / scale) ** 2)
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones_like(d)
    return w


def _weighted_median(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[good]
    weights = weights[good]
    if len(values) == 0:
        return np.nan
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    return float(values[np.searchsorted(cdf, 0.5 * cdf[-1], side="left")])


def _local_pdf(z_grid, z_values, weights, sigma=0.04):
    z_values = np.asarray(z_values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    good = np.isfinite(z_values) & np.isfinite(weights) & (weights > 0)
    z_values = z_values[good]
    weights = weights[good]
    if len(z_values) == 0:
        return None
    sigma = max(float(sigma), 1e-6)
    diff = (np.asarray(z_grid, dtype=np.float64)[:, None] - z_values[None, :]) / sigma
    p = np.exp(-0.5 * diff * diff) @ weights
    p = np.maximum(p, 0.0)
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        return None
    return (p / total).astype(np.float32)


def _pdf_median(z_grid, pdf):
    if not _valid_pdf(pdf):
        return np.nan
    p = np.asarray(pdf, dtype=np.float64)
    cdf = np.cumsum(p)
    idx = int(np.searchsorted(cdf, 0.5 * cdf[-1], side="left"))
    idx = min(max(idx, 0), len(z_grid) - 1)
    return float(z_grid[idx])


def _build_local_lookup(model, use_gpu=True):
    """Build SPEC training objects grouped by (region, SOM, BMU)."""
    meta = model.get("metadata", {})
    X = meta.get("c4_training_X_spec")
    z = meta.get("c4_training_z_spec")
    if X is None or z is None:
        raise RuntimeError(
            "El modelo no contiene c4_training_X_spec/c4_training_z_spec. "
            "Ejecuta train_submission_C4.py para generar el modelo C-4."
        )

    X = np.asarray(X, dtype=np.float32)
    z = np.asarray(z, dtype=np.float64)
    valid = np.isfinite(z) & np.isfinite(X).all(axis=1)
    X = X[valid]
    z = z[valid]

    from preprocessing import create_training_regions
    regions_train = create_training_regions(X, z, cuts=2, overlap=0.80)
    if len(regions_train) != len(model["regions"]):
        raise RuntimeError(
            f"Número de regiones inconsistente: entrenamiento={len(regions_train)}, "
            f"modelo={len(model['regions'])}."
        )

    lookup = {}
    for rid, reg in enumerate(regions_train):
        for sid in (0, 1):
            weights = model["regions"][rid][f"som_{sid}"]["weights"]
            bmu = _bmu(reg["X"], weights, use_gpu=use_gpu)
            for neuron in np.unique(bmu):
                m = bmu == neuron
                lookup[(rid, sid, int(neuron))] = {
                    "X": np.asarray(reg["X"])[m],
                    "z": np.asarray(reg["z"], dtype=np.float64)[m],
                }
    return lookup


def _get_z_grid(model):
    edges = np.asarray(model["metadata"]["z_edges_global"], dtype=np.float64)
    return 0.5 * (edges[:-1] + edges[1:])


def predict_objects_C4(matches, model, regions, z_edges_global=None,
                        sigma_distance=0.5, k_neighbors=None,
                        pdf_sigma=None, highz_threshold=None,
                        f_highz_threshold=None, delta_threshold=None,
                        use_gpu=True):
    """C-4 prediction: C-3 baseline + local photometric SPEC selector."""
    base = predict_objects_C3(
        matches,
        regions,
        z_edges_global=z_edges_global,
        sigma_distance=sigma_distance,
    )

    meta = model.get("metadata", {})
    k = int(k_neighbors if k_neighbors is not None else meta.get("c4_k_neighbors", 20))
    sigma_pdf = float(pdf_sigma if pdf_sigma is not None else meta.get("c4_pdf_sigma", 0.04))
    hz = float(highz_threshold if highz_threshold is not None else meta.get("c4_highz_threshold", 1.5))
    f_thr = float(f_highz_threshold if f_highz_threshold is not None else meta.get("c4_f_highz_threshold", 0.50))
    d_thr = float(delta_threshold if delta_threshold is not None else meta.get("c4_delta_threshold", 0.05))

    z_grid = _get_z_grid(model)
    lookup = _build_local_lookup(model, use_gpu=use_gpu)

    X_test = matches.get("X_test")
    if X_test is None:
        raise RuntimeError("C-4 requiere matches['X_test'] para el selector local.")
    X_test = np.asarray(X_test, dtype=np.float32)

    z_pred = np.asarray(base["z"], dtype=np.float32).copy()
    pdf_out = np.asarray(base["pdf"], dtype=np.float32).copy()
    local_used = np.zeros(len(z_pred), dtype=bool)
    local_fhighz = np.full(len(z_pred), np.nan, dtype=np.float32)
    local_delta = np.full(len(z_pred), np.nan, dtype=np.float32)
    local_distance = np.full(len(z_pred), np.nan, dtype=np.float32)
    local_n = np.zeros(len(z_pred), dtype=np.int32)

    for i in range(len(z_pred)):
        key = (int(matches["region"][i]), int(matches["som_id"][i]), int(matches["bmu"][i]))
        group = lookup.get(key)
        if group is None or len(group["z"]) == 0:
            continue
        ii, dist = _nearest_indices(X_test[i], group["X"], k=k)
        if len(ii) == 0:
            continue
        zz = group["z"][ii]
        w = _gaussian_weights(dist)
        if w.sum() <= 0:
            continue
        frac_highz = float(np.sum(w * (zz >= hz)) / np.sum(w))
        local_median = _weighted_median(zz, w)
        delta = float(local_median - z_pred[i])
        local_fhighz[i] = frac_highz
        local_delta[i] = delta
        local_distance[i] = float(np.median(dist))
        local_n[i] = len(zz)
        if frac_highz >= f_thr and delta >= d_thr:
            p_local = _local_pdf(z_grid, zz, w, sigma=sigma_pdf)
            if _valid_pdf(p_local):
                z_local = _pdf_median(z_grid, p_local)
                if np.isfinite(z_local):
                    z_pred[i] = np.float32(z_local)
                    pdf_out[i] = p_local
                    local_used[i] = True

    base["z_c3"] = np.asarray(base["z"], dtype=np.float32).copy()
    base["pdf_c3"] = np.asarray(base["pdf"], dtype=np.float32).copy()
    base["z"] = z_pred
    base["pdf"] = pdf_out
    base["c4_local_used"] = local_used
    base["c4_f_highz"] = local_fhighz
    base["c4_delta_local"] = local_delta
    base["c4_median_knn_distance"] = local_distance
    base["c4_n_neighbors"] = local_n
    return base


def save_qp_submission(pdf, z_edges, object_id, filename):
    """
    Save PDFs in the LSST PZ Challenge QP format.
    """

    import qp

    pdf = np.asarray(pdf, dtype=np.float32)
    z_edges = np.asarray(z_edges, dtype=np.float32)
    object_id = np.asarray(object_id)

    # PDF shape: (N_objects, N_bins)
    if pdf.ndim != 2:
        raise ValueError(
            f"pdf debe ser 2D, recibido shape={pdf.shape}"
        )

    z_centers = 0.5 * (
        z_edges[:-1] + z_edges[1:]
    )

    if pdf.shape[1] != len(z_centers):
        raise ValueError(
            "El número de bins del PDF no coincide con "
            "el número de centros de z."
        )

    # Normalize PDFs
    norm = np.sum(pdf, axis=1, keepdims=True)

    valid = (
        np.isfinite(norm[:, 0])
        & (norm[:, 0] > 0)
    )

    pdf_out = np.zeros_like(pdf)

    pdf_out[valid] = (
        pdf[valid]
        / norm[valid]
    )

    # zmode
    zmode = np.full(
        len(pdf),
        np.nan,
        dtype=np.float32
    )

    if np.any(valid):
        zmode[valid] = z_centers[
            np.argmax(
                pdf_out[valid],
                axis=1
            )
        ]

    # Create QP ensemble using the same API
    # as the original cosom_inference.py
    ens = qp.interp.create_ensemble(
        xvals=z_centers,
        yvals=pdf_out
    )

    ens.set_ancil(
        {
            "object_id": object_id,
            "zmode": zmode
        }
    )

    ens.write_to(filename)

# Public prediction entry point for the final submission model.
def predict_objects(matches, model, regions=None, **kwargs):

    if regions is None:
        regions = model["regions"]

    return predict_objects_C4(
        matches,
        model,
        regions,
        z_edges_global=model["metadata"]["z_edges_global"],
        **kwargs
    )
