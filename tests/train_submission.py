#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 13:35:00 2026

@author: alvaro
"""

"""
Training pipeline for Co-SOM submission.

C-4 keeps the C-3 training/PDF construction unchanged and adds the
photometric-local SPEC selector information needed at inference.

This module trains the Co-SOM model using an HDF5 training file
and stores a reduced inference model.
"""


import argparse
import pickle
import numpy as np


from preprocessing import (
    preprocess_train_hdf5,
    create_training_regions
)


from cosom_region import (
    train_region
)



# ==========================================================
# Parameters
# ==========================================================


params = {


    # SOM

    "points_per_neuron": 50,

    "learning_rate_0": 0.7,

    "epochs": 200,


    # Co-training

    "rounds": 15,

    "beta": 0.1,

    "m": 0.5,

    "threshold": 0.95,


    # Label fraction

    "labeled_fraction": 0.95,


    # GPU

    "batch_size": 65536,

    "mini_batch_size": 2048,

    # Experiment C-uncertainty: COSMOS many-band redshifts affect only final
    # neuron PDFs. Their contribution is gated by local COSMOS redshift
    # uncertainty and by the amount of existing SPEC/Co-SOM label support.
    "cosmos_pdf_weight": 0.10,

    # COSMOS weak-label uncertainty experiment. The uncertainty is estimated
    # locally from the scatter of COSMOS many-band redshifts per neuron.
    "cosmos_uncertainty_scale": 0.10,
    "cosmos_min_base_samples": 100,
    # Minimum distance in redshift from the base SPEC/Co-SOM support before
    # COSMOS is allowed to extend the final neuron PDF.
    "cosmos_tail_margin": 0.05

}



# ==========================================================
# Main training function
# ==========================================================


def train_submission(
        train_file,
        output_model,
        cosmos_pdf_weight=0.10,
        cosmos_uncertainty_scale=0.10,
        cosmos_min_base_samples=100,
        cosmos_tail_margin=0.05,
        seed=42
):
    """
    Train Co-SOM model for a challenge training file.

    Parameters
    ----------
    train_file : str or Path
        Input HDF5 training catalog.

    output_model : str or Path
        Output pickle model file.

    cosmos_pdf_weight : float
        Per-object multiplier for COSMOS-only labels used ONLY to build the
        final neuron PDFs. The SOM maps and co-training are otherwise unchanged.

    cosmos_uncertainty_scale : float
        Redshift-scatter scale used to downweight uncertain COSMOS labels.

    cosmos_min_base_samples : int
        Number of base SPEC/Co-SOM labels per neuron at which the COSMOS
        contribution is fully suppressed.

    seed : int
        Seed applied to NumPy and CuPy random generators for more comparable
        weight experiments.
    """

    cosmos_pdf_weight = float(cosmos_pdf_weight)
    if not np.isfinite(cosmos_pdf_weight) or cosmos_pdf_weight < 0.0:
        raise ValueError("--cosmos-pdf-weight must be finite and >= 0.")

    cosmos_uncertainty_scale = float(cosmos_uncertainty_scale)
    cosmos_min_base_samples = int(cosmos_min_base_samples)
    cosmos_tail_margin = float(cosmos_tail_margin)
    if not np.isfinite(cosmos_uncertainty_scale) or cosmos_uncertainty_scale <= 0.0:
        raise ValueError("--cosmos-uncertainty-scale must be finite and > 0.")
    if cosmos_min_base_samples < 0:
        raise ValueError("--cosmos-min-base-samples must be >= 0.")
    if not np.isfinite(cosmos_tail_margin) or cosmos_tail_margin < 0.0:
        raise ValueError("--cosmos-tail-margin must be finite and >= 0.")

    params["cosmos_pdf_weight"] = cosmos_pdf_weight
    params["cosmos_uncertainty_scale"] = cosmos_uncertainty_scale
    params["cosmos_min_base_samples"] = cosmos_min_base_samples
    params["cosmos_tail_margin"] = cosmos_tail_margin
    np.random.seed(int(seed))




    (
        X_train,
        z_train,
        object_id,
        preprocess_params,
        X_unlabeled,
        z_manyband_unlabeled,
        object_id_unlabeled

    ) = preprocess_train_hdf5(
        train_file,
        return_unlabeled=True
    )


    # Experiment C-uncertainty: COSMOS-only photometry does NOT enter the
    # co-training
    # unlabeled pool. Calibrated COSMOS many-band redshifts are used only as
    # weak labels when building the final neuron PDFs.








    # ------------------------------------------------------
    # Global redshift grid
    # ------------------------------------------------------

    finite_weak_z = z_manyband_unlabeled[np.isfinite(z_manyband_unlabeled)]
    z_grid_min = float(z_train.min())
    z_grid_max = float(z_train.max())
    if finite_weak_z.size:
        z_grid_min = min(z_grid_min, float(finite_weak_z.min()))
        z_grid_max = max(z_grid_max, float(finite_weak_z.max()))

    # ------------------------------------------------------
    # Global redshift grid
    # ------------------------------------------------------

    finite_weak_z = z_manyband_unlabeled[
        np.isfinite(z_manyband_unlabeled)
    ]

    z_grid_min = float(z_train.min())
    z_grid_max = float(z_train.max())

    if finite_weak_z.size:
        z_grid_min = min(
            z_grid_min,
            float(finite_weak_z.min())
        )
        z_grid_max = max(
            z_grid_max,
            float(finite_weak_z.max())
        )

    # ------------------------------------------------------
    # Final PDF resolution
    # ------------------------------------------------------
    # Use dz = 0.02 to reduce the final PDF size.
    # This gives approximately 300 bins over the full
    # redshift range while preserving the C-4 training
    # and inference methodology.

    PDF_DZ = 0.02

    n_bins = int(
        np.ceil(
            (z_grid_max - z_grid_min)
            / PDF_DZ
        )
    )

    z_grid_max_final = (
        z_grid_min
        + n_bins * PDF_DZ
    )

    params["z_edges_global"] = (
        z_grid_min
        + np.arange(
            n_bins + 1,
            dtype=np.float32
        ) * PDF_DZ
    )

    print(
        f"Global PDF grid: "
        f"dz={PDF_DZ:.3f}, "
        f"bins={n_bins}, "
        f"range=["
        f"{z_grid_min:.6f}, "
        f"{z_grid_max_final:.6f}"
        f"]"
    )


    # ------------------------------------------------------
    # Create training regions
    # ------------------------------------------------------



    regions = create_training_regions(

        X_train,

        z_train,

        cuts=2,

        overlap=0.80

    )



    params["n_regions"] = len(regions)



    # ------------------------------------------------------
    # Train each region
    # ------------------------------------------------------

    submission_regions = []


    for region in regions:







        params["region_id"] = (
            region["index"]
        )

        params["region_lower"] = (
            region["lower"]
        )

        params["region_upper"] = (
            region["upper"]
        )



        # Select COSMOS weak-label samples for this region only.
        # They are NOT added to the Co-SOM unlabeled pool.
        if region["index"] == len(regions) - 1:
            weak_mask = (
                np.isfinite(z_manyband_unlabeled)
                & (z_manyband_unlabeled >= region["lower"])
            )
        else:
            weak_mask = (
                np.isfinite(z_manyband_unlabeled)
                & (z_manyband_unlabeled >= region["lower"])
                & (z_manyband_unlabeled < region["upper"])
            )

        X_region_weak = X_unlabeled[weak_mask]
        z_region_weak = z_manyband_unlabeled[weak_mask]


        model_region = train_region(
            region["X"],
            region["z"],
            params,
            X_weak_cosmos=X_region_weak,
            z_weak_cosmos=z_region_weak
        )


        submission_regions.append(
        model_region
        )



    # ------------------------------------------------------
    # Build final model
    # ------------------------------------------------------

    model = {


        "regions":
            submission_regions,


        "metadata": {


            "preprocess_params":
                preprocess_params,


            "z_edges_global":
                params["z_edges_global"],

            "experiment": "C4_local_photometric_SPEC_selector",
            "cosmos_manyband_used_for_pretraining_or_cotraining_labels": False,
            "cosmos_only_used_as_unlabeled_photometry": False,
            "cosmos_manyband_used_as_weak_pdf_label": True,
            "cosmos_pdf_weight": float(params["cosmos_pdf_weight"]),
            "cosmos_uncertainty_scale": float(params["cosmos_uncertainty_scale"]),
            "cosmos_min_base_samples": int(params["cosmos_min_base_samples"]),
            "cosmos_tail_margin": float(params["cosmos_tail_margin"]),
            "cosmos_uncertainty_mode": "local_neuron_redshift_scatter",
            "cosmos_manyband_median_offset": float(preprocess_params.get("cosmos_manyband_median_offset", 0.0)),
            "cosmos_manyband_calibration_pairs": int(preprocess_params.get("cosmos_manyband_calibration_pairs", 0)),
            "point_estimate": "median_of_final_weighted_neuron_PDF",
            "gpu_som_stream_training": True,

            # --------------------------------------------------
            # C-4 local photometric selector
            # Training itself is unchanged from C-3. We retain
            # the preprocessed spectroscopic training set so
            # inference can inspect photometrically similar objects
            # inside the BMU that receives the validation/test object.
            # --------------------------------------------------
            "c4_local_selector": True,
            "c4_k_neighbors": 20,
            "c4_pdf_sigma": 0.04,
            "c4_highz_threshold": 1.5,
            "c4_f_highz_threshold": 0.50,
            "c4_delta_threshold": 0.05,
            "c4_store_training_features": True,
            "c4_store_training_redshifts": True,
            "c4_training_X_spec": np.asarray(X_train, dtype=np.float32),
            "c4_training_z_spec": np.asarray(z_train, dtype=np.float32)

        }


    }



    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    with open(
        output_model,
        "wb"
    ) as f:


        pickle.dump(
            model,
            f
        )





    return output_model





# ==========================================================
# Local test
# ==========================================================


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train Co-SOM with uncertainty-weighted COSMOS weak labels."
    )
    parser.add_argument(
        "--train",
        default="pz_challenge_taskset_3_cardinal_training_1yr.hdf5",
        help="80%% HDF5 training catalog."
    )
    parser.add_argument(
        "--output",
        default="pz_challenge_taskset_3_cardinal_pz_model_1yr.pkl",
        help="Output model pickle path. Use a distinct name for each weight."
    )
    parser.add_argument(
        "--cosmos-pdf-weight",
        type=float,
        default=0.10,
        help="Base contribution weight for COSMOS-only labels in final PDFs (>=0)."
    )
    parser.add_argument(
        "--cosmos-uncertainty-scale",
        type=float,
        default=0.10,
        help="Redshift-scatter scale controlling uncertainty downweighting (default: 0.10)."
    )
    parser.add_argument(
        "--cosmos-min-base-samples",
        type=int,
        default=100,
        help="Base SPEC/Co-SOM labels per neuron at which COSMOS is fully suppressed."
    )
    parser.add_argument(
        "--cosmos-tail-margin",
        type=float,
        default=0.05,
        help="Minimum redshift distance outside the base neuron support before COSMOS is added (default: 0.05)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for NumPy/CuPy random generators; keep fixed across comparisons."
    )
    args = parser.parse_args()

    train_submission(
        train_file=args.train,
        output_model=args.output,
        cosmos_pdf_weight=args.cosmos_pdf_weight,
        cosmos_uncertainty_scale=args.cosmos_uncertainty_scale,
        cosmos_min_base_samples=args.cosmos_min_base_samples,
        cosmos_tail_margin=args.cosmos_tail_margin,
        seed=args.seed,
    )
