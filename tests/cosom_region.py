#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 17:55:31 2026

@author: alvaro
"""

import math
import numpy as np
#import cupy as cp
#import gpu_som_3 as SOM


def initialize_region(X, z, params):
    """
    Initialize all data structures required to train one Co-SOM region.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (N_samples, N_features).

    z : np.ndarray
        Spectroscopic redshift labels.

    params : dict
        Dictionary containing all Co-SOM hyperparameters.

    Returns
    -------
    state : dict
        Dictionary containing all variables required during training.
    """

    # ==========================================================
    # Build training matrix
    # ==========================================================

    data = np.column_stack(
        (X, z)
    ).astype(np.float32)

    n_samples = data.shape[0]
    n_features = X.shape[1]

    # ==========================================================
    # Compute SOM map size
    # ==========================================================

    points_per_neuron = params["points_per_neuron"]

    ratio = 4 / 3

    n_neurons = math.ceil(n_samples / points_per_neuron)

    size_x = math.ceil(
        math.sqrt(n_neurons * ratio)
    )

    size_y = math.ceil(
        n_neurons / size_x
    )
    # ==========================================================
    # SOM training scales derived from map size
    # ==========================================================

    map_size = max(
        size_x,
        size_y
    )

    sigma_0 = 0.5 * map_size

    step = 0.1 * sigma_0

    # ==========================================================
    # SOM params
    # ==========================================================

    matrix_numbers = np.arange(
        size_x * size_y,
        dtype=int
    ).reshape(size_x, size_y)

    matrix_ind = SOM.index_matrix(
        size_x,
        size_y
    ).astype(int)

    xx, yy = SOM.get_coordinates(
        size_x,
        size_y
    )

    neighbors_map = SOM.all_neighbors(
        matrix_numbers
    )

    # ==========================================================
    # Copy SOM params to GPU
    # ==========================================================

    matrix_ind_gpu = cp.asarray(matrix_ind)

    xx_gpu = cp.asarray(xx.flatten())

    yy_gpu = cp.asarray(yy.flatten())

    neighbors_map_gpu = cp.asarray(neighbors_map)

    # ==========================================================
    # Initialize SOM weights
    # ==========================================================

    weights_gpu_1 = cp.asarray(

        np.random.rand(
            size_x * size_y,
            n_features

        ).astype(np.float32)

    )

    weights_gpu_2 = cp.asarray(

        np.random.rand(
            size_x * size_y,
            n_features

        ).astype(np.float32)

    )
    print("labeled_fraction =", params["labeled_fraction"])

    # ==========================================================
    # Split labeled and unlabeled samples
    # ==========================================================

    unlabeled_index, initial_label_index = SOM.stratified_sampling(
        data,
        n_features,
        sample_size=params["labeled_fraction"]

    )

    # The original 5% spectroscopic holdout remains unlabeled.
    unlabeled_set = data[unlabeled_index].astype(np.float32)
    unlabeled_source = np.full(
        len(unlabeled_set), "hidden_spectroscopic", dtype="<U24"
    )

    # COSMOS-only photometry is NOT part of the co-training unlabeled pool.
    # It is used only later as a weak-label source when constructing the
    # final neuron PDFs. The unlabeled pool contains only hidden
    # spectroscopic objects from this redshift region.
    n_cosmos_unlabeled = 0

    print("Hidden spectroscopic pool:", int(np.sum(unlabeled_source == "hidden_spectroscopic")))
    print("COSMOS-only added to unlabeled pool:", n_cosmos_unlabeled)
    print("Total co-training unlabeled pool:", len(unlabeled_set))

    mu = np.zeros(len(unlabeled_set), dtype=np.float32)

    data_aux = data[
        initial_label_index
    ]

    tset1_index, tset2_index = SOM.stratified_sampling(

        data_aux,
        n_features,
        sample_size=0.5

    )

    tset_1 = data_aux[
        tset1_index
    ].astype(np.float32)

    tset_2 = data_aux[
       tset2_index
    ].astype(np.float32)

    # ==========================================================
    # Training state
    # ==========================================================

    state = {
        "n_neurons": size_x * size_y,

        "region_id": params["region_id"],
        
        "region_lower": params["region_lower"],

        "region_upper": params["region_upper"],

        "z_edges_global": params["z_edges_global"],

        "data": data,
        
        "sigma_0": sigma_0,

        "step": step,
        
        "features": n_features,

        "size_x": size_x,

        "size_y": size_y,

        "weights_gpu_1": weights_gpu_1,

        "weights_gpu_2": weights_gpu_2,

        "tset_1": tset_1,

        "tset_2": tset_2,

        "unlabeled_set": unlabeled_set,

        "unlabeled_source": unlabeled_source,
        "n_unlabeled": unlabeled_set.shape[0],
        "n_cosmos_unlabeled_initial": n_cosmos_unlabeled,
        "cosmos_pseudo_labeled_count": 0,
        "hidden_spec_pseudo_labeled_count": 0,

        "mu": mu,

        "matrix_ind_gpu": matrix_ind_gpu,

        "xx_gpu": xx_gpu,

        "yy_gpu": yy_gpu,

        "neighbors_map_gpu": neighbors_map_gpu

    }

    return state

def pretrain_region(state, params):
    """
    Initial independent training of the two SOMs in one Co-SOM region.

    Each SOM is pretrained using its corresponding labeled subset.
    No pseudo-labeling or co-training is performed here.

    Parameters
    ----------
    state : dict
        State dictionary returned by initialize_region().

    params : dict
        Training hyperparameters.

    Returns
    -------
    state : dict
        Updated state with pretrained SOM weights.
    """

    # ==========================================================
    # Extract variables
    # ==========================================================

    n_features = state["features"]

    matrix_ind_gpu = state["matrix_ind_gpu"]
    xx_gpu = state["xx_gpu"]
    yy_gpu = state["yy_gpu"]

    # Current weights
    weights_gpu_1 = state["weights_gpu_1"]
    weights_gpu_2 = state["weights_gpu_2"]

    # Labeled datasets
    tset_1 = state["tset_1"]
    tset_2 = state["tset_2"]
    
    sigma_0 = state["sigma_0"]

    step = state["step"]


    # ======================================================
    # CUDA streams: crear una sola vez
    # ======================================================

    stream_1 = cp.cuda.Stream(non_blocking=True)
    stream_2 = cp.cuda.Stream(non_blocking=True)
    
    # ==========================================================
    # Remove redshift column
    # SOM only receives photometric features
    # ==========================================================

    X_train_1 = tset_1[:, :n_features]

    X_train_2 = tset_2[:, :n_features]

    #inicio = time.time()
    # ==========================================================
    # Pretrain SOM 1
    # ==========================================================
    with stream_1:
        weights_gpu_1 = SOM.train_minibatch_som(
            X_train_1,
            state["size_x"],
            state["size_y"],
            sigma_0,
            params["learning_rate_0"],
            params["epochs"],
            weights_gpu_1,
            step,
            matrix_ind_gpu,
            xx_gpu,
            yy_gpu,
            batch_size=params["batch_size"],
            mini_batch_size=params["mini_batch_size"],
            stream=stream_1
        )

    # ==========================================================
    # Pretrain SOM 2
    # ==========================================================
    with stream_2:
        weights_gpu_2 = SOM.train_minibatch_som(
            X_train_2,
            state["size_x"],
            state["size_y"],
            sigma_0,
            params["learning_rate_0"],
            params["epochs"],
            weights_gpu_2,
            step,
            matrix_ind_gpu,
            xx_gpu,
            yy_gpu,
            batch_size=params["batch_size"],
            mini_batch_size=params["mini_batch_size"],
            stream=stream_2
        )
    stream_1.synchronize()
    stream_2.synchronize()

    #fin = time.time()

    #print(
    #    f"Tiempo pretrain SOM-1 + SOM-2 paralelo: "
    #    f"{fin - inicio:.4f} s"
    #)
    # ==========================================================
    # Update state
    # ==========================================================

    state["weights_gpu_1"] = weights_gpu_1

    state["weights_gpu_2"] = weights_gpu_2


    return state

def co_training_region(state, params):
    """
    Perform iterative Co-SOM training for one region.

    At each iteration:
    1. Retrain both SOMs using the current labeled sets.
    2. Assign neuron redshift labels.
    3. Select reliable pseudo-labeled galaxies.
    4. Transfer selected galaxies between SOM training sets.

    PDF estimation is intentionally excluded and performed only
    after the complete co-training process.

    Parameters
    ----------
    state : dict
        State dictionary from initialize_region().

    params : dict
        Co-training and SOM parameters.

    Returns
    -------
    state : dict
        Updated state after co-training.
    """

    print("Starting Co-SOM training")


    # ==========================================================
    # Extract variables
    # ==========================================================

    weights_gpu_1 = state["weights_gpu_1"]
    weights_gpu_2 = state["weights_gpu_2"]

    tset_1 = state["tset_1"]
    tset_2 = state["tset_2"]

    unlabeled_set = state["unlabeled_set"]
    unlabeled_source = state["unlabeled_source"]

    mu = state["mu"]

    FEATURES = state["features"]

    matrix_ind_gpu = state["matrix_ind_gpu"]
    xx_gpu = state["xx_gpu"]
    yy_gpu = state["yy_gpu"]
    neighbors_gpu = state["neighbors_map_gpu"]

    sigma = state["sigma_0"]

    step = state["step"]

    # ======================================================
    # CUDA streams: crear una sola vez
    # ======================================================

    stream_1 = cp.cuda.Stream(non_blocking=True)
    stream_2 = cp.cuda.Stream(non_blocking=True)

    
    # ==========================================================
    # Co-training loop
    # ==========================================================

    for j in range(params["rounds"]):

        print(f"Co-training iteration {j+1}/{params['rounds']}")
        # ======================================================
        # Train SOM-1 and SOM-2 in parallel
        # ======================================================

        #inicio = time.time()

        with stream_1:
            weights_gpu_1 = SOM.train_minibatch_som(
                tset_1[:, :FEATURES],
                state["size_x"],
                state["size_y"],
                sigma,
                params["learning_rate_0"],
                params["epochs"],
                weights_gpu_1,
                step,
                matrix_ind_gpu,
                xx_gpu,
                yy_gpu,
                batch_size=params["batch_size"],
                mini_batch_size=params["mini_batch_size"],
                stream=stream_1
            )

        with stream_2:
            weights_gpu_2 = SOM.train_minibatch_som(
                tset_2[:, :FEATURES],
                state["size_x"],
                state["size_y"],
                sigma,
                params["learning_rate_0"],
                params["epochs"],
                weights_gpu_2,
                step,
                matrix_ind_gpu,
                xx_gpu,
                yy_gpu,
                batch_size=params["batch_size"],
                mini_batch_size=params["mini_batch_size"],
                stream=stream_2
            )

        # Esperar a que ambas SOM terminen
        stream_1.synchronize()
        stream_2.synchronize()
        
        #fin = time.time()
        
        #print("Tiempo SOM-1 + SOM-2 paralelo:", fin - inicio)

        # ======================================================
        # Label assignment
        # ======================================================
        pseudo_labels_1, _, _, _ = SOM.label_assignment(
            tset_1,
            weights_gpu_1,
            FEATURES
        )


        pseudo_labels_2, _, _, _ = SOM.label_assignment(
            tset_2,
            weights_gpu_2,
            FEATURES
        )

        # ======================================================
        # Split unlabeled pool
        # ======================================================

        perm = np.random.permutation(
            unlabeled_set.shape[0]
        )

        c1 = int(len(perm)*0.5)
        c2 = int(len(perm)*0.9)

        train_ind_1 = perm[:c1]
        train_ind_2 = perm[c1:c2]


        # ======================================================
        # SOM-1 labels for SOM-2
        # ======================================================
        #inicio = time.time()
        pseudo_prediction_1, index_pseudo_labels_1 = SOM.sample_selection(

            unlabeled_set,
            weights_gpu_1,
            pseudo_labels_1,

            train_ind_1,

            mu,

            params["beta"],
            params["m"],
            params["threshold"],

            FEATURES,

            neighbors_gpu
        )


        # ======================================================
        # SOM-2 labels for SOM-1
        # ======================================================

        pseudo_prediction_2, index_pseudo_labels_2 = SOM.sample_selection(

            unlabeled_set,
            weights_gpu_2,
            pseudo_labels_2,

            train_ind_2,

            mu,

            params["beta"],
            params["m"],
            params["threshold"],

            FEATURES,

            neighbors_gpu
        )
        #fin = time.time()
        #tiempo_total = fin - inicio
        #print(tiempo_total)
        #print(s)


        # ======================================================
        # Update training sets
        # ======================================================

        if index_pseudo_labels_1.size > 0:

            new_rows = unlabeled_set[
                index_pseudo_labels_1
            ].copy()

            new_rows[:, FEATURES] = pseudo_prediction_1

            tset_2 = np.vstack(
                [
                    tset_2,
                    new_rows
                ]
            )


        if index_pseudo_labels_2.size > 0:

            new_rows = unlabeled_set[
                index_pseudo_labels_2
            ].copy()

            new_rows[:, FEATURES] = pseudo_prediction_2

            tset_1 = np.vstack(
                [
                    tset_1,
                    new_rows
                ]
            )


        # ======================================================
        # Remove selected samples from pool
        # ======================================================

        remove_idx = np.concatenate(
            [
                index_pseudo_labels_1,
                index_pseudo_labels_2
            ]
        )


        if remove_idx.size > 0:

            # Record whether accepted pseudo-labels came from held-out spectra
            # or COSMOS-only photometry. This is diagnostic only; both sources
            # are treated as unlabeled until selected by the algorithm.
            selected_sources = unlabeled_source[remove_idx]
            state["cosmos_pseudo_labeled_count"] += int(
                np.sum(selected_sources == "COSMOS-only")
            )
            state["hidden_spec_pseudo_labeled_count"] += int(
                np.sum(selected_sources == "hidden_spectroscopic")
            )

            unlabeled_set = np.delete(unlabeled_set, remove_idx, axis=0)
            unlabeled_source = np.delete(unlabeled_source, remove_idx, axis=0)
            mu = np.delete(mu, remove_idx, axis=0)


        # ======================================================
        # Update sigma
        # ======================================================

        sigma = max(sigma - step, 0.1)


        # Free GPU temporary memory

        cp.get_default_memory_pool().free_all_blocks()



    # ==========================================================
    # Save final state
    # ==========================================================

    state["weights_gpu_1"] = weights_gpu_1
    state["weights_gpu_2"] = weights_gpu_2

    state["tset_1"] = tset_1
    state["tset_2"] = tset_2

    state["unlabeled_set"] = unlabeled_set
    state["unlabeled_source"] = unlabeled_source
    state["mu"] = mu
    state["sigma_final"] = sigma

    print("Pseudo-labeled COSMOS-only objects:", state["cosmos_pseudo_labeled_count"])
    print("Pseudo-labeled hidden spectroscopic objects:", state["hidden_spec_pseudo_labeled_count"])
    print("Unlabeled objects remaining:", len(unlabeled_set))

    return state

def _pdf_median_per_neuron(pdf, z_centers):
    """Return one posterior median per neuron from a 2-D PDF table."""
    pdf = np.asarray(pdf, dtype=np.float64)
    centers = np.asarray(z_centers, dtype=np.float64).ravel()
    if pdf.ndim != 2 or pdf.shape[1] != centers.size:
        raise ValueError("PDF table and redshift-bin centers have incompatible shapes.")
    medians = np.full(pdf.shape[0], np.nan, dtype=np.float32)
    for i, row in enumerate(pdf):
        row = np.where(np.isfinite(row) & (row > 0.0), row, 0.0)
        total = row.sum()
        if total > 0.0:
            idx = int(np.searchsorted(np.cumsum(row / total), 0.5, side="left"))
            medians[i] = np.float32(centers[min(idx, centers.size - 1)])
    return medians


def _region_weak_cosmos_data(X, z, params):
    """Select finite weak labels for this region using spectroscopic limits.

    Non-final regions use [lower, upper); the last region accepts every weak
    redshift >= lower, extending the PDF support beyond the spectroscopic max.
    This routing is used only for weak-PDF construction, not SOM training.
    """
    if X is None or z is None:
        return None
    X = np.asarray(X, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32).ravel()
    if X.ndim != 2 or X.shape[0] != z.size:
        raise ValueError("X_weak_cosmos and z_weak_cosmos must have matching rows.")
    valid = np.isfinite(z) & np.isfinite(X).all(axis=1)
    lower = float(params.get("region_lower", -np.inf))
    upper = float(params.get("region_upper", np.inf))
    rid = int(params.get("region_id", 0))
    n_regions = int(params.get("n_regions", 2))
    if rid == n_regions - 1:
        mask = valid & (z >= lower)
    else:
        mask = valid & (z >= lower) & (z < upper)
    if not np.any(mask):
        return None
    return np.column_stack((X[mask], z[mask])).astype(np.float32, copy=False)


def _combine_base_and_weak_pdf_inputs(
    base_bmu, base_z_gpu, base_weights,
    weak_data, som_weights_gpu, n_features,
    cosmos_pdf_weight,
    cosmos_uncertainty_scale=0.10,
    cosmos_min_base_samples=100,
    cosmos_tail_margin=0.05,
):
    """
    Add COSMOS only as a weak extension outside the SPEC/Co-SOM PDF support.

    C-2 keeps the trained SPEC/Co-SOM PDF as the core calibration. COSMOS
    objects are admitted only when their redshift lies outside the local
    redshift support already represented by the base labels of the same
    neuron. The COSMOS contribution is additionally reduced by empirical
    local redshift scatter and by the amount of existing base-label support.

    This is intentionally a PDF-construction change only: SOM training,
    pretraining and co-training are unchanged.
    """
    if weak_data is None or len(weak_data) == 0 or cosmos_pdf_weight <= 0.0:
        return base_bmu, base_z_gpu, base_weights, 0

    _, weak_bmu, weak_weights, weak_z_gpu = SOM.label_assignment(
        weak_data, som_weights_gpu, n_features
    )

    n_neurons = int(som_weights_gpu.shape[0])
    base_bmu_i = base_bmu.astype(cp.int32)
    base_z = base_z_gpu.astype(cp.float64)
    weak_bmu_i = weak_bmu.astype(cp.int32)
    weak_z = weak_z_gpu.astype(cp.float64)

    # ----------------------------------------------------------
    # Base-label support per neuron.
    # ----------------------------------------------------------
    base_counts = cp.bincount(
        base_bmu_i, minlength=n_neurons
    ).astype(cp.float64)

    if cosmos_min_base_samples <= 0:
        support_factor = cp.ones_like(base_counts)
    else:
        support_factor = cp.clip(
            1.0 - base_counts / float(cosmos_min_base_samples),
            0.0,
            1.0
        )

    # ----------------------------------------------------------
    # Local SPEC/Co-SOM redshift support [min, max] per neuron.
    # Neurons without base labels have no protected core, so all
    # COSMOS support is eligible (subject to uncertainty).
    # ----------------------------------------------------------
    base_z_min = cp.full(n_neurons, cp.inf, dtype=cp.float64)
    base_z_max = cp.full(n_neurons, -cp.inf, dtype=cp.float64)
    cp.minimum.at(base_z_min, base_bmu_i, base_z)
    cp.maximum.at(base_z_max, base_bmu_i, base_z)

    margin = max(float(cosmos_tail_margin), 0.0)

    tail_mask = (
        (base_counts[weak_bmu_i] <= 0.0)
        | (weak_z < (base_z_min[weak_bmu_i] - margin))
        | (weak_z > (base_z_max[weak_bmu_i] + margin))
    )

    # ----------------------------------------------------------
    # Empirical COSMOS uncertainty per neuron.
    # ----------------------------------------------------------
    weak_counts = cp.bincount(
        weak_bmu_i, minlength=n_neurons
    ).astype(cp.float64)

    z_sum = cp.bincount(
        weak_bmu_i, weights=weak_z, minlength=n_neurons
    ).astype(cp.float64)

    z_sq_sum = cp.bincount(
        weak_bmu_i, weights=weak_z * weak_z, minlength=n_neurons
    ).astype(cp.float64)

    global_mean = cp.mean(weak_z)
    global_var = cp.mean((weak_z - global_mean) ** 2)
    global_sigma = cp.sqrt(cp.maximum(global_var, 0.0))

    safe_counts = cp.maximum(weak_counts, 1.0)
    local_mean = z_sum / safe_counts
    local_var = cp.maximum(
        z_sq_sum / safe_counts - local_mean * local_mean,
        0.0
    )
    local_sigma = cp.sqrt(local_var)
    reliable_local = weak_counts >= 3.0
    local_sigma = cp.where(reliable_local, local_sigma, global_sigma)

    sigma_ref = max(float(cosmos_uncertainty_scale), 1.0e-6)
    uncertainty_factor = 1.0 / (
        1.0 + (local_sigma / sigma_ref) ** 2
    )

    # ----------------------------------------------------------
    # COSMOS is now used only outside the base PDF support.
    # ----------------------------------------------------------
    object_support_factor = support_factor[weak_bmu_i]
    object_uncertainty_factor = uncertainty_factor[weak_bmu_i]

    weak_weights = weak_weights.astype(cp.float64)
    weak_weights = (
        weak_weights
        * float(cosmos_pdf_weight)
        * object_support_factor
        * object_uncertainty_factor
        * tail_mask.astype(cp.float64)
    )

    keep_mask = weak_weights > 0.0
    weak_bmu_keep = weak_bmu[keep_mask]
    weak_z_keep = weak_z_gpu[keep_mask]
    weak_weights_keep = weak_weights[keep_mask]
    n_kept = int(cp.count_nonzero(keep_mask).get())

    #print(
    #    f"COSMOS empirical uncertainty: global sigma={float(global_sigma.get()):.4f}; "
    #    f"sigma_ref={sigma_ref:.4f}"
    #)
    #print(
    #    "C-2 COSMOS tail support: "
    #    f"only outside base PDF support by margin={margin:.3f}"
    #)
    #print(
    #    "COSMOS support protection: "
    #    f"full suppression at >= {int(cosmos_min_base_samples)} base labels/neuron"
    #)
    #print(
    #    "COSMOS tail objects admitted to final PDFs: "
    #    f"{n_kept}/{len(weak_data)}"
    #)
    #if n_kept > 0:
        #print(
        #    "COSMOS effective tail PDF weight range: "
        #    f"{float(cp.min(weak_weights_keep).get()):.6g} -> "
        #    f"{float(cp.max(weak_weights_keep).get()):.6g}"
    #    )

    if n_kept == 0:
        return base_bmu, base_z_gpu, base_weights, 0

    return (
        cp.concatenate((base_bmu, weak_bmu_keep)),
        cp.concatenate((base_z_gpu, weak_z_keep)),
        cp.concatenate((base_weights, weak_weights_keep)),
        n_kept,
    )



def generate_region_pdf(
    state,
    params,
    X_weak_cosmos=None,
    z_weak_cosmos=None,
):
    """
    Generate final neuron PDFs after Co-SOM training.

    Experiment C keeps the pointwise spectroscopic/pseudo-label medians
    separate, but adds calibrated COSMOS-only redshifts to each neuron PDF
    with a sample-weight multiplier. COSMOS-only photometry does not enter
    SOM pretraining or co-training.

    This function is executed only once after co-training finishes.

    Parameters
    ----------
    state : dict
        Final Co-SOM state after co-training.

    params : dict
        Model parameters.

    Returns
    -------
    state : dict
        Updated state containing complete SOM inference models.
    """

    # ==========================================================
    # Extract variables
    # ==========================================================

    weights_gpu_1 = state["weights_gpu_1"]
    weights_gpu_2 = state["weights_gpu_2"]

    tset_1 = state["tset_1"]
    tset_2 = state["tset_2"]

    FEATURES = state["features"]

    n_neurons = state["n_neurons"]
    cosmos_pdf_weight = float(params.get("cosmos_pdf_weight", 0.10))
    cosmos_uncertainty_scale = float(params.get("cosmos_uncertainty_scale", 0.10))
    cosmos_min_base_samples = int(params.get("cosmos_min_base_samples", 100))
    cosmos_tail_margin = float(params.get("cosmos_tail_margin", 0.05))
    if not np.isfinite(cosmos_pdf_weight) or cosmos_pdf_weight < 0.0:
        raise ValueError("cosmos_pdf_weight must be finite and >= 0.")
    if not np.isfinite(cosmos_uncertainty_scale) or cosmos_uncertainty_scale <= 0.0:
        raise ValueError("cosmos_uncertainty_scale must be finite and > 0.")
    if cosmos_min_base_samples < 0:
        raise ValueError("cosmos_min_base_samples must be >= 0.")
    if not np.isfinite(cosmos_tail_margin) or cosmos_tail_margin < 0.0:
        raise ValueError("cosmos_tail_margin must be finite and >= 0.")
    weak_data = _region_weak_cosmos_data(X_weak_cosmos, z_weak_cosmos, params)
    #weak_sample_count = 0 if weak_data is None else int(len(weak_data))
    #print(
    #    f"Region {state['region_id']}: weak COSMOS labels for PDFs="
    #    f"{weak_sample_count}; per-object weight={cosmos_pdf_weight:.3f}"
    #)

    # ==========================================================
    # Final label assignment SOM-1 (base samples only)
    # ==========================================================

    (
        z_median_1,
        bmu_idx_1,
        weights_c1,
        z_gpu_1

    ) = SOM.label_assignment(
        tset_1,
        weights_gpu_1,
        FEATURES
    )


    # ==========================================================
    # Final PDF SOM-1
    # ==========================================================

    bmu_pdf_1, z_pdf_gpu_1, weights_pdf_1, weak_count_1 = (
        _combine_base_and_weak_pdf_inputs(
            bmu_idx_1, z_gpu_1, weights_c1,
            weak_data, weights_gpu_1, FEATURES,
            cosmos_pdf_weight,
            cosmos_uncertainty_scale,
            cosmos_min_base_samples,
            cosmos_tail_margin,
        )
    )
    (
        pdf_1,
        _,
        z_centers_1
    ) = SOM.build_neuron_pdf(
        bmu_pdf_1,
        z_pdf_gpu_1,
        weights_pdf_1,
        n_neurons,
        state["z_edges_global"]
    )
    # C point estimates are the medians of the final weighted PDFs so weak
    # COSMOS labels affect predictions, while A/B's training-label medians
    # would otherwise remain unchanged.
    z_median_pdf_1 = _pdf_median_per_neuron(pdf_1, z_centers_1)


    # ==========================================================
    # Final label assignment SOM-2
    # ==========================================================

    (
        z_median_2,
        bmu_idx_2,
        weights_c2,
        z_gpu_2

    ) = SOM.label_assignment(
        tset_2,
        weights_gpu_2,
        FEATURES
    )


    # ==========================================================
    # Final PDF SOM-2
    # ==========================================================

    bmu_pdf_2, z_pdf_gpu_2, weights_pdf_2, weak_count_2 = (
        _combine_base_and_weak_pdf_inputs(
            bmu_idx_2, z_gpu_2, weights_c2,
            weak_data, weights_gpu_2, FEATURES,
            cosmos_pdf_weight,
            cosmos_uncertainty_scale,
            cosmos_min_base_samples,
            cosmos_tail_margin,
        )
    )
    (
        pdf_2,
        _,
        z_centers_2
    ) = SOM.build_neuron_pdf(
        bmu_pdf_2,
        z_pdf_gpu_2,
        weights_pdf_2,
        n_neurons,
        state["z_edges_global"]
    )
    z_median_pdf_2 = _pdf_median_per_neuron(pdf_2, z_centers_2)


    # ==========================================================
    # Build final SOM inference models
    # ==========================================================

    state["som_0"] = {

       "weights": cp.asnumpy(weights_gpu_1),

       "n_neurons": n_neurons,
       
       "n_features": state["features"],
       
       "size_x": state["size_x"],

       "size_y": state["size_y"],

       "z_median": z_median_pdf_1,

       "z_median_base_labels": z_median_1,

       "pdf": pdf_1,

    }


    state["som_1"] = {

       "weights": cp.asnumpy(weights_gpu_2),

       "n_neurons": n_neurons,
       
       "n_features": state["features"],
       
       "size_x": state["size_x"],

       "size_y": state["size_y"],

       "z_median": z_median_pdf_2,

       "z_median_base_labels": z_median_2,

       "pdf": pdf_2,

    }
    
    state["region"] = {
        "id": state["region_id"],
        "lower": state["region_lower"],
        "upper": state["region_upper"],
        "cosmos_weak_pdf_samples": int(max(weak_count_1, weak_count_2)),
        "cosmos_pdf_weight": cosmos_pdf_weight,
        "z_median_source": "median_of_final_weighted_pdf",
    }


    # ==========================================================
    # Clean temporary GPU arrays
    # ==========================================================

    del weights_c1
    del weights_c2
    del z_gpu_1
    del z_gpu_2
    del bmu_pdf_1, z_pdf_gpu_1, weights_pdf_1
    del bmu_pdf_2, z_pdf_gpu_2, weights_pdf_2

    cp.get_default_memory_pool().free_all_blocks()


    return state

def extract_region_model(state):
    """
    Extract only the information required for Co-SOM inference.

    Removes training-only variables such as:
    training data, co-training sets,
    GPU temporary arrays, and diagnostics.

    Parameters
    ----------
    state : dict
        Complete trained Co-SOM state.

    Returns
    -------
    model : dict
        Lightweight inference model.
    """
    model = {

         "region": {

              "id": state["region_id"],

              "lower": state["region_lower"],

              "upper": state["region_upper"]

         },

         "som_0": state["som_0"],

         "som_1": state["som_1"]

    }      


    return model

def train_region(
    X,
    z,
    params,
    X_weak_cosmos=None,
    z_weak_cosmos=None,
):
    """
    Complete Co-SOM training pipeline for one redshift region.

    Executes:
    
    1. Region initialization
    2. Independent SOM pretraining
    3. Iterative co-training
    4. Final neuron PDF generation
    5. Extraction of inference model

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.

    z : np.ndarray
        Spectroscopic redshifts.

    params : dict
        Co-SOM parameters.

    Returns
    -------
    model : dict
        Lightweight Co-SOM inference model.
    """


    # ==========================================================
    # Initialize region
    # ==========================================================

    state = initialize_region(
        X,
        z,
        params
    )


    # ==========================================================
    # Initial SOM pretraining
    # ==========================================================

    state = pretrain_region(
        state,
        params
    )


    # ==========================================================
    # Co-training iterations
    # ==========================================================

    state = co_training_region(
        state,
        params
    )


    # ==========================================================
    # Final PDF generation
    # ==========================================================

    state = generate_region_pdf(
        state,
        params,
        X_weak_cosmos=X_weak_cosmos,
        z_weak_cosmos=z_weak_cosmos,
    )


    # ==========================================================
    # Extract inference-only model
    # ==========================================================

    model = extract_region_model(
        state
    )


    return model
'''
        # ======================================================
        # Train SOM-1
        # ======================================================
        inicio = time.time()
        weights_gpu_1 = SOM.train_minibatch_som(

            tset_1[:, :FEATURES],

            state["size_x"],
            state["size_y"],

            sigma,
            params["learning_rate_0"],
            params["epochs"],

            weights_gpu_1,

            step,

            matrix_ind_gpu,
            xx_gpu,
            yy_gpu,

            batch_size=params["batch_size"],
            mini_batch_size=params["mini_batch_size"]
        )
        fin = time.time()
        tiempo_total = fin - inicio
        print(tiempo_total)



        # ======================================================
        # Train SOM-2
        # ======================================================
        inicio = time.time()
        weights_gpu_2 = SOM.train_minibatch_som(

            tset_2[:, :FEATURES],

            state["size_x"],
            state["size_y"],

            sigma,
            params["learning_rate_0"],
            params["epochs"],

            weights_gpu_2,

            step,

            matrix_ind_gpu,
            xx_gpu,
            yy_gpu,

            batch_size=params["batch_size"],
            mini_batch_size=params["mini_batch_size"]
        )
        fin = time.time()
        tiempo_total = fin - inicio
        print(tiempo_total)
'''
