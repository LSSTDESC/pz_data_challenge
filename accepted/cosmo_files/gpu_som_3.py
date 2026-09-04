
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 19:26:11 2025

@author: alvaro
"""
import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import gaussian_filter1d

def index_matrix(size_x, size_y):
    """
    Generate a linear index-to-coordinate mapping for a SOM.

    This function creates a grid of neuron coordinates and flattens it into
    a 2D array where each row corresponds to a neuron and contains its
    (x, y) position in the grid.

    Parameters
    ----------
    size_x : int
        Number of neurons along the x-axis (rows of the SOM grid).

    size_y : int
        Number of neurons along the y-axis (columns of the SOM grid).

    Returns
    -------
    matrix_ind : np.ndarray of shape (size_x * size_y, 2), dtype=np.int32
        Array where each row corresponds to a neuron and stores its
        2D coordinates in the SOM grid as:
        [x_coordinate, y_coordinate]

        The neurons are ordered in row-major format (flattened grid).
    """

    # Create a 2D grid of coordinates
    # xx contains row indices, yy contains column indices
    xx, yy = np.meshgrid(
        np.arange(size_x),
        np.arange(size_y),
        indexing='ij'
    )

    # Flatten the grid into a list of neuron coordinates
    matrix_ind = np.column_stack((xx.ravel(), yy.ravel()))

    return matrix_ind.astype(np.int32)  

def get_coordinates(size_x, size_y):
    """
    Generate the x and y coordinate matrices for the SOM lattice.

    These matrices are used during training to efficiently compute
    topological distances between neurons and the Best Matching Unit (BMU)
    without repeatedly reconstructing neuron coordinates.

    Parameters
    ----------
    size_x : int
        Number of neurons along the x-axis (rows).

    size_y : int
        Number of neurons along the y-axis (columns).

    Returns
    -------
    xx : np.ndarray of shape (size_x * size_y,), dtype=np.float32
        Flattened x-coordinates of every neuron.

    yy : np.ndarray of shape (size_x * size_y,), dtype=np.float32
        Flattened y-coordinates of every neuron.

    Notes
    -----
    The y-coordinates are shifted by half a cell on alternating rows in
    order to represent a staggered (hexagonal-like) SOM topology.
    """

    xx, yy = np.meshgrid(
        np.arange(size_x, dtype=np.float32),
        np.arange(size_y, dtype=np.float32),
        indexing="ij"
    )

    yy[::2] -= 0.5

    return xx.ravel(), yy.ravel()

def all_neighbors(index_matrix):
    """
    Compute the fixed topological neighbors for a staggered
    hexagonal-like Self-Organizing Map.

    Each neuron can have up to six neighbors. Boundary neurons have
    fewer neighbors, and the remaining entries are filled with -1.

    Parameters
    ----------
    index_matrix : np.ndarray of shape (size_x, size_y)
        Matrix containing the linear index of every neuron.

    Returns
    -------
    neighbors : np.ndarray of shape (n_neurons, 6), dtype=np.int32
        Neighbor lookup table.

        Each row corresponds to one neuron and contains the indices of
        its neighboring neurons. Missing neighbors are represented by -1.
    """

    rows, cols = index_matrix.shape

    neighbors = -np.ones((rows * cols, 6), dtype=np.int32)

    for i in range(rows):
        for j in range(cols):

            current = index_matrix[i, j]
            neighs = []

            if i % 2 == 0:

                if i > 0:
                    neighs.append(index_matrix[i - 1, j])

                    if j > 0:
                        neighs.append(index_matrix[i - 1, j - 1])

                if i < rows - 1:

                    if j > 0:
                        neighs.append(index_matrix[i + 1, j - 1])

                    neighs.append(index_matrix[i + 1, j])

            else:

                if i > 0:
                    neighs.append(index_matrix[i - 1, j])

                    if j < cols - 1:
                        neighs.append(index_matrix[i - 1, j + 1])

                if i < rows - 1:

                    neighs.append(index_matrix[i + 1, j])

                    if j < cols - 1:
                        neighs.append(index_matrix[i + 1, j + 1])

            if j > 0:
                neighs.append(index_matrix[i, j - 1])

            if j < cols - 1:
                neighs.append(index_matrix[i, j + 1])

            neighbors[current, :len(neighs)] = neighs

    return neighbors

def train_minibatch_som(data, size_x, size_y, sigma_0, learning_rate_0, epochs,
                        weights_gpu, step, matrix_ind_gpu, xx_gpu, yy_gpu, batch_size=65536,
                        mini_batch_size=2048):
    """
    GPU Mini-Batch Self-Organizing Map (SOM) training.

    This implementation performs SOM training entirely on the GPU using
    hierarchical batching. The dataset is first divided into large batches,
    which are subsequently processed as smaller mini-batches. Weight updates
    are accumulated over each mini-batch and applied once per iteration,
    reducing kernel launch overhead while preserving the SOM learning dynamics.

    Euclidean distances are computed using the inner-product formulation:

        ||x - w||² = ||x||² + ||w||² - 2 x·w

    where the squared norms of both the dataset and the neuron weights are
    precomputed whenever possible to reduce redundant operations.

    Parameters
    ----------
    data : ndarray
        Training samples.
    size_x, size_y : int
        SOM dimensions.
    sigma_0 : float
        Initial neighborhood radius.
    learning_rate_0 : float
        Initial learning rate.
    epochs : int
        Maximum number of training epochs.
    weights_gpu : cupy.ndarray
        Initial neuron weights stored on the GPU.
    step : float
        Minimum allowed neighborhood radius.
    matrix_ind_gpu : cupy.ndarray
        Mapping from neuron indices to 2-D coordinates.
    xx_gpu, yy_gpu : cupy.ndarray
        Precomputed neuron coordinate grids.
    batch_size : int, optional
        Size of the outer training batch.
    mini_batch_size : int, optional
        Size of each mini-batch used for weight updates.

    Returns
    -------
    weights_gpu : cupy.ndarray
        Trained SOM weights.

    """

    # ============================================================
    # Dataset
    # ============================================================
    data_gpu = cp.asarray(data, dtype=cp.float64)
    N = data_gpu.shape[0]
    data_norm = cp.sum(data_gpu * data_gpu, axis=1, keepdims=True)

    # ============================================================
    # Weight initialization
    # ============================================================
    weights_gpu = weights_gpu.astype(cp.float64)
    copy_weights = weights_gpu.copy()
    weight_norm = cp.sum(weights_gpu * weights_gpu, axis=1)[None, :]

    # ============================================================
    # Training loop
    # ============================================================
    for i in range(epochs):

        learning_rate = learning_rate_0 * cp.exp(-i / epochs)

        sigma = sigma_0 * cp.exp(
            -i / (epochs / cp.log(sigma_0))
        )

        sigma = cp.maximum(sigma, step)

        perm = cp.random.permutation(N)

        # ========================================================
        # Large batches
        # ========================================================
        for start in range(0, N, batch_size):

            end = min(start + batch_size, N)

            idx_batch = perm[start:end]

            batch = data_gpu[idx_batch]
            batch_norm_full = data_norm[idx_batch]

            # ====================================================
            # Mini-batches
            # ====================================================
            for mb_start in range(0, batch.shape[0], mini_batch_size):

                mb_end = min(mb_start + mini_batch_size, batch.shape[0])

                mini_batch = batch[mb_start:mb_end]
                mini_batch_norm = batch_norm_full[mb_start:mb_end]

                # =================================================
                # Squared Euclidean distances
                # =================================================
                dists = (
                    mini_batch_norm
                    + weight_norm
                    - 2 * (mini_batch @ weights_gpu.T)
                )

                BMU_idx = cp.argmin(dists, axis=1)
                BMU_coords = matrix_ind_gpu[BMU_idx]

                # =================================================
                # Gaussian neighborhood
                # =================================================
                dx = xx_gpu[None, :] - BMU_coords[:, 0, None]
                dy = yy_gpu[None, :] - BMU_coords[:, 1, None]

                dist2_map = dx * dx + dy * dy

                hck = cp.exp(
                    -dist2_map / (2.0 * sigma * sigma)
                )

                # =================================================
                # Mini-batch weight update
                # =================================================
                delta = (
                    learning_rate
                    * hck[:, :, None]
                    * (
                        mini_batch[:, None, :]
                        - weights_gpu[None, :, :]
                    )
                )

                delta_mean = cp.mean(delta, axis=0)

                weights_gpu += delta_mean

                weight_norm = cp.sum(
                    weights_gpu * weights_gpu,
                    axis=1
                )[None, :]

        # ========================================================
        # Convergence criterion
        # ========================================================
        umbral = cp.mean(
            cp.abs(weights_gpu - copy_weights)
        )

        if umbral < 1e-5:
            break

        copy_weights = weights_gpu.copy()


    return weights_gpu


def label_assignment(data, weights_gpu, features,
                     batch_size=65536):
    """
    Assign robust redshift labels to SOM neurons.

    Each input sample is mapped to its Best Matching Unit (BMU). A robust
    median redshift and Median Absolute Deviation (MAD) are then computed
    independently for every neuron. The MAD is converted into a robust
    standard deviation estimate and used to derive Cauchy weights for each
    galaxy.

    Parameters
    ----------
    data : ndarray
        Training catalog containing observables and spectroscopic redshifts.
    weights_gpu : cupy.ndarray
        SOM neuron weights.
    features : int
        Number of input features.
    batch_size : int, optional
        Number of samples processed simultaneously when searching BMUs.

    Returns
    -------
    z_median : ndarray
        Robust median redshift assigned to every neuron.
    bmu_idx : cupy.ndarray
        BMU index of every training sample.
    weights : cupy.ndarray
        Robust Cauchy weight assigned to every training sample.
    z_gpu : cupy.ndarray
        Redshift values stored on the GPU.
    """

    N_samples = data.shape[0]
    N_neurons = weights_gpu.shape[0]

    # ============================================================
    # Dataset
    # ============================================================
    data_gpu = cp.asarray(data[:, :features], dtype=cp.float32)
    z_gpu = cp.asarray(data[:, features], dtype=cp.float32)

    # ============================================================
    # BMU computation
    # ============================================================
    bmu_idx = cp.zeros(N_samples, dtype=cp.int32)

    w2 = cp.sum(weights_gpu * weights_gpu, axis=1)
    weights_T = weights_gpu.T

    for start in range(0, N_samples, batch_size):

        end = min(start + batch_size, N_samples)

        batch = data_gpu[start:end]

        x2 = cp.sum(batch * batch, axis=1, keepdims=True)

        dists = cp.maximum(
            x2 + w2 - 2 * (batch @ weights_T),
            0.0
        )

        bmu_idx[start:end] = cp.argmin(dists, axis=1)

    # ============================================================
    # Robust neuron statistics
    # ============================================================
    z_median = cp.zeros(N_neurons, dtype=cp.float32)
    mad_per_neuron = cp.full(N_neurons, 1e-6, dtype=cp.float32)

    for i in range(N_neurons):

        mask = (bmu_idx == i)

        if cp.any(mask):

            z_local = z_gpu[mask]

            median = cp.median(z_local)

            z_median[i] = median

            mad_per_neuron[i] = cp.median(
                cp.abs(z_local - median)
            )

    mad_per_neuron *= 1.4826
    mad_per_neuron = cp.clip(mad_per_neuron, 1e-6, None)

    abs_dev = cp.abs(z_gpu - z_median[bmu_idx])

    mad = mad_per_neuron[bmu_idx]

    c = 2.0

    r = abs_dev / (c * mad)

    weights = 1.0 / (1.0 + r * r)

    return (
        cp.asnumpy(z_median),
        bmu_idx,
        weights,
        z_gpu,
    )


def build_neuron_pdf(
        bmu_idx,
        z_gpu,
        weights,
        n_neurons,
        z_edges_global):
    """
    Build weighted redshift probability density functions (PDFs)
    for every SOM neuron using a common global redshift grid.

    A weighted redshift histogram is constructed for each neuron using
    the robust Cauchy weights obtained during label assignment. The
    histograms are normalized, Gaussian-smoothed and renormalized to
    obtain the final neuron PDFs.

    Parameters
    ----------
    bmu_idx : cupy.ndarray
        BMU index of every training sample.

    z_gpu : cupy.ndarray
        Spectroscopic redshifts stored on the GPU.

    weights : cupy.ndarray
        Robust Cauchy weights for every training sample.

    n_neurons : int
        Number of SOM neurons.

    z_edges_global : ndarray
        Common redshift bin edges used by all SOMs and regions.

    Returns
    -------
    pdf : ndarray
        Probability density function for every neuron.

    z_edges : ndarray
        Redshift bin edges.

    z_centers : ndarray
        Redshift bin centers.
    """


    # ==========================================================
    # Histogram bins
    # ==========================================================

    bins = cp.asarray(
        z_edges_global,
        dtype=cp.float32
    )

    n_bins = len(z_edges_global) - 1


    # ==========================================================
    # Assign redshift bins
    # ==========================================================

    z_bin_index = cp.digitize(
        z_gpu,
        bins
    ) - 1

    z_bin_index = cp.clip(
        z_bin_index,
        0,
        n_bins - 1
    )


    # ==========================================================
    # Weighted histogram
    # ==========================================================

    hist = cp.zeros(
        (n_neurons, n_bins),
        dtype=cp.float32
    )


    cp.add.at(
        hist,
        (bmu_idx, z_bin_index),
        weights
    )


    occupancy = hist.sum(axis=1)


    norm = cp.where(
        occupancy[:, None] == 0,
        1,
        occupancy[:, None]
    )

    pdf = hist / norm


    # ==========================================================
    # Gaussian smoothing
    # ==========================================================

    pdf = gaussian_filter1d(
        pdf,
        sigma=1.0,
        axis=1
    )


    norm = cp.sum(
        pdf,
        axis=1,
        keepdims=True
    )

    norm = cp.where(
        norm == 0,
        1,
        norm
    )

    pdf /= norm


    # ==========================================================
    # Empty neurons
    # ==========================================================

    pdf[occupancy == 0] = 0


    # ==========================================================
    # Output grid
    # ==========================================================

    z_edges = np.asarray(
        z_edges_global
    )

    z_centers = (
        z_edges[:-1]
        +
        z_edges[1:]
    ) / 2


    return (
        cp.asnumpy(pdf),
        z_edges,
        z_centers
    )

def sample_selection(unlabeled_set, w_train_gpu, pseudo_labels, sample_ind,
                     mu_c, beta, m, threshold, FEATURES, neighbors_gpu,
                     batch_size=65536):
    """
    Selección de muestras vectorizada en GPU por lotes, con mu_c en GPU para evitar
    problemas de broadcasting y conversiones CPU-GPU.
    Maneja el caso en que no haya índices seleccionados para filtrar.
    """
    # Pasar datos a GPU
    unlabeled_gpu = cp.asarray(unlabeled_set[:, :FEATURES], dtype=cp.float32)
    pseudo_labels_gpu = cp.asarray(pseudo_labels)
    mu_gpu = cp.asarray(mu_c)

    # Trabajar solo con los índices de sample_ind
    unlabeled_sel_gpu = unlabeled_gpu[sample_ind, :]
    N = unlabeled_sel_gpu.shape[0]

    # --- BMU por muestra (batch) ---
    bmu_idx_parts = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        diff = unlabeled_sel_gpu[start:end, None, :] - w_train_gpu[None, :, :]
        dist = cp.linalg.norm(diff, axis=2)
        bmu_idx_parts.append(cp.argmin(dist, axis=1))
        del diff, dist

    bmu_idx = cp.concatenate(bmu_idx_parts, axis=0)
    pseudo_prediction_gpu = pseudo_labels_gpu[bmu_idx]

    # --- Distancias Du y Dl ---
    bmu_neighbors = neighbors_gpu[bmu_idx]
    neighbor_labels = pseudo_labels_gpu[bmu_neighbors]
    valid_mask = neighbor_labels != 0

    Du = cp.sum(cp.abs(neighbor_labels - pseudo_prediction_gpu[:, None]) * valid_mask, axis=1)
    avg_labels = cp.sum(neighbor_labels * valid_mask, axis=1) / cp.sum(valid_mask, axis=1)
    Dl = cp.sum(cp.abs(neighbor_labels - avg_labels[:, None]) * valid_mask, axis=1)
    Dx = Du + beta * Dl  # len(Dx) == len(sample_ind)

    # --- Actualización de mu en GPU ---
    mu_gpu[sample_ind] = (1 - m) * mu_gpu[sample_ind] + m * Dx

    # --- Probabilidades ---
    max_mu = cp.max(mu_gpu[sample_ind])
    den_mu = cp.sum(max_mu - mu_gpu[sample_ind])
    p = cp.zeros(unlabeled_set.shape[0], dtype=cp.float32)
    if den_mu == 0:
        p[:] = 1.0 / unlabeled_set.shape[0]
    else:
        p[sample_ind] = (max_mu - mu_gpu[sample_ind]) / den_mu

    max_prob = cp.max(p)
    index_pseudo_labels = cp.where(p >= max_prob * threshold)[0]

    # --- Filtrado por distancia al BMU más cercano (batch) ---
    min_dist_parts = []
    for start in range(0, len(index_pseudo_labels), batch_size):
        end = min(start + batch_size, len(index_pseudo_labels))
        idx_batch = index_pseudo_labels[start:end]
        diff_select = unlabeled_gpu[idx_batch, None, :] - w_train_gpu[None, :, :]
        dist_select = cp.linalg.norm(diff_select, axis=2)
        min_dist_parts.append(cp.min(dist_select, axis=1))
        del diff_select, dist_select

    # Manejo del caso vacío
    if len(min_dist_parts) > 0:
        min_dist = cp.concatenate(min_dist_parts, axis=0)
        final_mask = min_dist <= min_dist.min() * 1.5
        final_index = cp.asnumpy(index_pseudo_labels[final_mask])
        pseudo_prediction_final = pseudo_labels_gpu[final_index].get()
    else:
        final_index = np.array([], dtype=int)
        pseudo_prediction_final = np.array([], dtype=np.float32)

    # --- Liberar memoria temporal GPU ---
    del unlabeled_gpu, pseudo_labels_gpu, unlabeled_sel_gpu
    del bmu_idx, pseudo_prediction_gpu
    del bmu_neighbors, neighbor_labels, valid_mask
    del Du, avg_labels, Dl, Dx, min_dist_parts, p
    cp.get_default_memory_pool().free_all_blocks()

    # --- Pasar mu de vuelta a CPU al final ---
    mu_c[:] = cp.asnumpy(mu_gpu)

    return pseudo_prediction_final, final_index

def stratified_sampling(sample,feature,splits=10,sample_size=0.1):
    ranges = np.zeros((splits))
    counts = np.zeros((sample.shape[0]))
    range_sample = np.amax(sample[:,feature])-np.amin(sample[:,feature])
    step = range_sample /splits
    start = np.amin(sample[:,feature])
    for i in range (splits):
        if i == splits-1:
            index = np.asarray(np.where(sample[:,feature]>=start)).flatten()
        else:
            index = np.asarray(np.where((sample[:,feature]>=start)&
                            (sample[:,feature]<start+step))).flatten()
        ranges[i]=index.shape[0]
        counts[index] = i+1
        start += step
    values = ranges/sample.shape[0]
    new_sample = np.empty((0),dtype=int)
    new_size = int(sample_size*sample.shape[0])
    for i in range (splits):
        proportion = int(new_size*values[i])
        index_ranges = np.asarray(np.where(counts==i+1)).flatten()
        if index_ranges.shape[0] != 0:
            aux = np.random.choice(np.arange(index_ranges.shape[0]),size = proportion,replace=False)
            new_sample = np.append(new_sample,index_ranges[aux])
    index=np.arange(sample.shape[0] )
    return np.setdiff1d(index,new_sample),new_sample