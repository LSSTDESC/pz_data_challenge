#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 17:55:31 2026

@author: alvaro
"""

import math
import numpy as np
import cupy as cp
import gpu_som_3 as SOM


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

    unlabeled_set = data[
        unlabeled_index
    ].astype(np.float32)

    mu = np.zeros(
        unlabeled_set.shape[0],
        dtype=np.float32
    )

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
        
        "n_unlabeled": unlabeled_set.shape[0],

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


    # ==========================================================
    # Remove redshift column
    # SOM only receives photometric features
    # ==========================================================

    X_train_1 = tset_1[:, :n_features]

    X_train_2 = tset_2[:, :n_features]


    # ==========================================================
    # Pretrain SOM 1
    # ==========================================================

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

        batch_size=params.get(
            "batch_size",
            65536
        ),

        mini_batch_size=params.get(
            "mini_batch_size",
            2048
        )
    )


    # ==========================================================
    # Pretrain SOM 2
    # ==========================================================

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

        batch_size=params.get(
            "batch_size",
            65536
        ),

        mini_batch_size=params.get(
            "mini_batch_size",
            2048
        )
    )


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

    mu = state["mu"]

    FEATURES = state["features"]

    matrix_ind_gpu = state["matrix_ind_gpu"]
    xx_gpu = state["xx_gpu"]
    yy_gpu = state["yy_gpu"]
    neighbors_gpu = state["neighbors_map_gpu"]

    sigma = state["sigma_0"]

    step = state["step"]


    # ==========================================================
    # Co-training loop
    # ==========================================================

    for j in range(params["rounds"]):

        print(f"Co-training iteration {j+1}/{params['rounds']}")


        # ======================================================
        # Train SOM-1
        # ======================================================

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


        # ======================================================
        # Train SOM-2
        # ======================================================

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

            unlabeled_set = np.delete(
                unlabeled_set,
                remove_idx,
                axis=0
            )

            mu = np.delete(
                mu,
                remove_idx,
                axis=0
            )


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
    state["mu"] = mu
    state["sigma_final"] = sigma
    
    return state

def generate_region_pdf(state, params):
    """
    Generate final redshift probability density functions (PDFs)
    after the complete Co-SOM training process.

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

    # ==========================================================
    # Final label assignment SOM-1
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

    (
        pdf_1,
        _,
        _

    ) = SOM.build_neuron_pdf(
        bmu_idx_1,
        z_gpu_1,
        weights_c1,
        n_neurons,
        state["z_edges_global"]
    )


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

    (
        pdf_2,
        _,
        _

    ) = SOM.build_neuron_pdf(
        bmu_idx_2,
        z_gpu_2,
        weights_c2,
        n_neurons,
        state["z_edges_global"]
    )


    # ==========================================================
    # Build final SOM inference models
    # ==========================================================

    state["som_0"] = {

       "weights": cp.asnumpy(weights_gpu_1),

       "n_neurons": n_neurons,
       
       "n_features": state["features"],
       
       "size_x": state["size_x"],

       "size_y": state["size_y"],

       "z_median": z_median_1,

       "pdf": pdf_1,

    }


    state["som_1"] = {

       "weights": cp.asnumpy(weights_gpu_2),

       "n_neurons": n_neurons,
       
       "n_features": state["features"],
       
       "size_x": state["size_x"],

       "size_y": state["size_y"],

       "z_median": z_median_2,

       "pdf": pdf_2,

    }
    
    state["region"] = {

    "id": state["region_id"],

    "lower": state["region_lower"],

    "upper": state["region_upper"]

    }


    # ==========================================================
    # Clean temporary GPU arrays
    # ==========================================================

    del weights_c1
    del weights_c2
    del z_gpu_1
    del z_gpu_2

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

def train_region(X, z, params):
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
        params
    )


    # ==========================================================
    # Extract inference-only model
    # ==========================================================

    model = extract_region_model(
        state
    )


    return model