#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:46:46 2026

@author: alvaro
"""

import os
import pickle
import numpy as np
import cupy as cp
import pandas as pd
from confidence import estimate_confidence
import qp

def load_models(model_dir):
    """
    Load all trained Co-SOM models and metadata.

    This function loads:

    - Global metadata.
    - Global confidence model.
    - All trained Co-SOM region models.

    Parameters
    ----------
    model_dir : str
        Directory containing the trained models.

    Returns
    -------
    models : dict
        Dictionary containing all information required during
        inference.

        metadata : dict
            Global preprocessing parameters and redshift grid.

        confidence : dict
            Trained confidence model.

        regions : list of dict
            List containing all trained Co-SOM regions sorted by
            region identifier.
    """

    # ==========================================================
    # Load metadata
    # ==========================================================

    with open(
        os.path.join(
            model_dir,
            "metadata.pkl"
        ),
        "rb"
    ) as f:

        metadata = pickle.load(f)


    # ==========================================================
    # Load confidence model
    # ==========================================================

    with open(
        os.path.join(
            model_dir,
            "confidence.pkl"
        ),
        "rb"
    ) as f:

        confidence = pickle.load(f)


    # ==========================================================
    # Load all trained regions
    # ==========================================================

    regions = []

    n_regions = metadata["n_regions"]

    for region_id in range(n_regions):

        filename = os.path.join(

            model_dir,

            f"region_{region_id}.pkl"

        )

        with open(filename, "rb") as f:

            regions.append(
                pickle.load(f)
            )


    # ==========================================================
    # Build model container
    # ==========================================================

    models = {

        "metadata": metadata,

        "confidence": confidence,

        "regions": regions

    }


    return models


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
    # Move data to GPU
    # ==========================================================

    X_gpu = cp.asarray(
        X,
        dtype=cp.float32
    )


    # ==========================================================
    # Search all SOMs
    # ==========================================================

    for region_id, model in enumerate(models):

        for som_id in [0, 1]:


            som = model[
                f"som_{som_id}"
            ]


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


            # BMU in this SOM

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


            del weights_gpu
            del dists



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
            candidate_distance

    }

def predict_objects(matches, regions):
    """
    Retrieve redshift predictions and PDFs from selected SOM neurons.

    The closest BMU is checked first. If its PDF is empty or invalid,
    the next closest BMU candidates are tested until a valid PDF is found.

    Parameters
    ----------
    matches : dict
        Output from find_best_bmu_cpu().

    regions : list
        Reduced Co-SOM submission models.

    Returns
    -------
    predictions : dict
        Predicted redshift information and PDFs.
    """


    n_objects = len(
        matches["bmu"]
    )


    # ==========================================================
    # Allocate outputs
    # ==========================================================

    z_pred = np.zeros(
        n_objects,
        dtype=np.float32
    )


    pdf_list = []


    used_region = np.zeros(
        n_objects,
        dtype=np.int32
    )


    used_som = np.zeros(
        n_objects,
        dtype=np.int32
    )


    used_bmu = np.zeros(
        n_objects,
        dtype=np.int32
    )


    fallback_counter = 0



    # ==========================================================
    # Search valid PDF
    # ==========================================================

    n_candidates = (
        matches["bmu_candidates"].shape[1]
    )


    for i in range(n_objects):


        selected_pdf = None


        for j in range(n_candidates):


            region = matches[
                "region_candidates"
            ][i, j]


            som_id = matches[
                "som_candidates"
            ][i, j]


            bmu = matches[
                "bmu_candidates"
            ][i, j]


            som = regions[region][
                f"som_{som_id}"
            ]


            pdf_candidate = som[
                "pdf"
            ][bmu]



            # ----------------------------------------------
            # Check PDF validity
            # ----------------------------------------------

            if (
                np.all(
                    np.isfinite(pdf_candidate)
                )
                and
                np.sum(pdf_candidate) > 0
            ):


                selected_pdf = pdf_candidate


                used_region[i] = region

                used_som[i] = som_id

                used_bmu[i] = bmu


                if j > 0:

                    fallback_counter += 1


                break



        # ==================================================
        # Emergency fallback
        # ==================================================

        if selected_pdf is None:


            region = matches["region"][i]

            som_id = matches["som_id"][i]

            bmu = matches["bmu"][i]


            som = regions[region][
                f"som_{som_id}"
            ]


            n_bins = som[
                "pdf"
            ].shape[1]


            selected_pdf = np.ones(
                n_bins,
                dtype=np.float32
            )


            selected_pdf /= n_bins


            used_region[i] = region

            used_som[i] = som_id

            used_bmu[i] = bmu



        # ==================================================
        # Point estimate from neuron
        # ==================================================

        som = regions[
            used_region[i]
        ][
            f"som_{used_som[i]}"
        ]


        z_pred[i] = som[
            "z_median"
        ][
            used_bmu[i]
        ]


        pdf_list.append(
            selected_pdf
        )



    # ==========================================================
    # Convert PDFs to array
    # ==========================================================

    pdf = np.asarray(
        pdf_list,
        dtype=np.float32
    )



    print(
        f"Fallback BMUs used: {fallback_counter} / {n_objects}"
    )



    return {


        # Final selected neuron

        "region":
            used_region,


        "som_id":
            used_som,


        "bmu":
            used_bmu,


        # Original closest BMU

        "bmu_original":
            matches["bmu"],


        "distance":
            matches["distance"],


        # Neuron point estimate

        "z":
            z_pred,


        # Object p(z)

        "pdf":
            pdf

    }

def add_prediction_confidence(
        X,
        predictions,
        confidence_model):
    """
    Add OOD confidence information to Co-SOM predictions.

    The confidence is estimated using the Isolation Forest model
    trained exclusively with the photometric feature distribution
    of the training sample.

    Parameters
    ----------
    X : np.ndarray
        Photometric features of the objects being predicted.

    predictions : dict
        Output dictionary from predict_objects().

        Expected keys:
            - region
            - som_id
            - bmu
            - distance
            - z
            - pdf

    confidence_model : dict
        Trained OOD confidence model generated by
        train_confidence_model().

    Returns
    -------
    predictions : dict
        Updated prediction dictionary including:

            confidence :
                Confidence percentile (0-100).

            confidence_category :
                Qualitative confidence label.

            ood_score :
                Raw Isolation Forest score.
    """


    # ==========================================================
    # Estimate OOD confidence
    # ==========================================================

    confidence, category, scores = estimate_confidence(

        X,

        confidence_model

    )


    # ==========================================================
    # Add information to predictions
    # ==========================================================

    predictions["confidence"] = confidence

    predictions["confidence_category"] = category

    predictions["ood_score"] = scores


    return predictions

def build_qp_output(predictions, object_id, z_edges):
    """
    Build final photo-z prediction output.

    Parameters
    ----------
    predictions : dict
        Output from predict_objects().

    object_id : ndarray
        Unique object identifier.

    z_edges : ndarray
        Global redshift bin edges.

    Returns
    -------
    output : dict
        Final prediction catalog.
    """


    output = {


        # Object ID

        "id":
            object_id,



        # Point estimate

        "z_phot":
            predictions["z"],



        # Histogram PDF

        "pdf":
            predictions["pdf"],



        # Selected SOM information

        "region":
            predictions["region"],


        "som_id":
            predictions["som_id"],


        "bmu":
            predictions["bmu"],



        # Original closest BMU before fallback

        "bmu_original":
            predictions["bmu_original"],



        # Distance to original BMU

        "distance":
            predictions["distance"],



        # OOD information

        "confidence":
            predictions["confidence"],


        "confidence_category":
            predictions["confidence_category"],


        "ood_score":
            predictions["ood_score"]

    }

    return output

def save_predictions_parquet(
        output,
        filename,
        z_edges):
    """
    Save final photo-z predictions in parquet format.

    PDFs are stored as histogram bins.
    The redshift grid is stored separately.

    Parameters
    ----------
    output : dict
        Output dictionary from build_qp_output().

    filename : str
        Output parquet filename.

    z_edges : ndarray
        Redshift bin edges.
    """


    # ==========================================================
    # Build main dataframe
    # ==========================================================

    df = pd.DataFrame({

        "id":
            output["id"],


        "z_phot":
            output["z_phot"],


        "distance":
            output["distance"],


        "confidence":
            output["confidence"],


        "confidence_category":
            output["confidence_category"],


        "ood_score":
            output["ood_score"],


        "region":
            output["region"],


        "som_id":
            output["som_id"],


        "bmu":
            output["bmu"]

    })


    # ==========================================================
    # Add PDF histogram bins
    # ==========================================================

    pdf = output["pdf"]


    pdf_columns = pd.DataFrame(
        pdf,
        columns=[
            f"pz_{i}"
            for i in range(pdf.shape[1])
        ],
        index=df.index
    )


    df = pd.concat(
        [
            df,
            pdf_columns
        ],
        axis=1
    )


    # ==========================================================
    # Save parquet catalog
    # ==========================================================

    df.to_parquet(
        filename,
        index=False
    )


    # ==========================================================
    # Save redshift grid
    # ==========================================================

    np.save(
        filename.replace(
            ".parquet",
            "_z_edges.npy"
        ),
        z_edges
    )


    print(
        f"Saved: {filename}"
    )

    print(
        "Shape:",
        df.shape
    )
#=============================================================================#
#                            pz_challenge                                     #
#=============================================================================#

def find_best_bmu_cpu(
        X,
        models,
        k_candidates=5
):
    """
    CPU BMU search for Co-SOM submission inference.

    Parameters
    ----------
    X : np.ndarray
        Test feature matrix.

    models : list
        Reduced submission Co-SOM models.

    k_candidates : int
        Number of BMU candidates.

    Returns
    -------
    matches : dict
        BMU information compatible with predict_objects().
    """


    N = X.shape[0]


    # ==========================================================
    # Initialize candidate storage
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


    X = X.astype(
        np.float32
    )


    # ==========================================================
    # Search all regions and SOMs
    # ==========================================================

    for region_id, model in enumerate(models):


        for som_id in [0,1]:


            som = model[
                f"som_{som_id}"
            ]


            weights = som[
                "weights"
            ].astype(
                np.float32
            )


            # ----------------------------------------------
            # Squared Euclidean distance
            # ----------------------------------------------

            x2 = np.sum(
                X * X,
                axis=1,
                keepdims=True
            )


            w2 = np.sum(
                weights * weights,
                axis=1
            )


            dists = np.maximum(

                x2
                +
                w2
                -
                2 * X @ weights.T,

                0.0

            )


            bmu = np.argmin(
                dists,
                axis=1
            )


            distance = np.sqrt(
                np.min(
                    dists,
                    axis=1
                )
            )


            # ==================================================
            # Merge with existing candidates
            # ==================================================

            new_distance = np.column_stack(
                [
                    candidate_distance,
                    distance
                ]
            )


            new_region = np.column_stack(
                [
                    candidate_region,
                    np.full(
                        N,
                        region_id,
                        dtype=np.int32
                    )
                ]
            )


            new_som = np.column_stack(
                [
                    candidate_som,
                    np.full(
                        N,
                        som_id,
                        dtype=np.int32
                    )
                ]
            )


            new_bmu = np.column_stack(
                [
                    candidate_bmu,
                    bmu
                ]
            )


            idx = np.argpartition(
                new_distance,
                k_candidates-1,
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
    # Sort final candidates
    # ==========================================================

    idx = np.argsort(
        candidate_distance,
        axis=1
    )


    rows = np.arange(N)[:,None]


    candidate_distance = (
        candidate_distance[
            rows,
            idx
        ]
    )


    candidate_region = (
        candidate_region[
            rows,
            idx
        ]
    )


    candidate_som = (
        candidate_som[
            rows,
            idx
        ]
    )


    candidate_bmu = (
        candidate_bmu[
            rows,
            idx
        ]
    )


    return {


        "region":
            candidate_region[:,0],


        "som_id":
            candidate_som[:,0],


        "bmu":
            candidate_bmu[:,0],


        "distance":
            candidate_distance[:,0],


        "bmu_candidates":
            candidate_bmu,


        "region_candidates":
            candidate_region,


        "som_candidates":
            candidate_som,


        "distance_candidates":
            candidate_distance

    }

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

def save_qp_submission(
        pdf,
        z_edges,
        object_id,
        filename
):
    """
    Save Co-SOM p(z) predictions in qp format.

    Parameters
    ----------
    pdf : np.ndarray
        Object probability density functions.

        Shape:
            (N_objects, N_z_bins)

    z_edges : np.ndarray
        Redshift bin edges.

        Shape:
            (N_z_bins + 1,)

    object_id : np.ndarray
        Object identifiers.

    filename : str or Path
        Output qp file.
    """


    # ==========================================================
    # Convert histogram bins to redshift centers
    # ==========================================================

    z_centers = (
        z_edges[:-1]
        +
        z_edges[1:]
    ) / 2



    # ==========================================================
    # Normalize PDFs
    #
    # qp expects probability densities:
    #
    # integral p(z) dz = 1
    #
    # ==========================================================

    dz = np.diff(
        z_centers
    )


    # If grid is uniform

    if np.allclose(
        dz,
        dz[0]
    ):

        dz = dz[0]


        norm = (
            np.sum(pdf, axis=1)
            *
            dz
        )


    else:

        norm = np.sum(
            pdf[:, :-1] * dz,
            axis=1
        )


    # Avoid division problems

    norm[norm == 0] = 1.0


    pdf = pdf / norm[:, None]



    # ==========================================================
    # Compute zmode from PDF
    # ==========================================================

    zmode = z_centers[
        np.argmax(
            pdf,
            axis=1
        )
    ]



    # ==========================================================
    # Create qp interpolated ensemble
    #
    # Each object has:
    #
    # p(z_i)
    #
    # ==========================================================
    
    ensemble = qp.interp.create_ensemble(
    xvals=z_centers,
    yvals=pdf
    )


    # ==========================================================
    # Add metadata
    # ==========================================================

    ensemble.set_ancil(
        {
            "object_id": object_id,
            "zmode": zmode
        }
    )



    # ==========================================================
    # Write qp file
    # ==========================================================

    ensemble.write_to(
        filename
    )


    print(
        "Saved qp file:",
        filename
    )