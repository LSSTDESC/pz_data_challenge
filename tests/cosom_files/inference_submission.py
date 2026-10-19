#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference pipeline for Co-SOM photo-z challenge.

Loads a trained C-4 Co-SOM model,
processes an HDF5 test catalog,
predicts p(z),
and saves the result in qp format.
"""

from pathlib import Path

from cosom_inference import (
    load_submission_model,
    find_best_bmu,
    predict_objects,
    save_qp_submission,
)

from preprocessing import (
    preprocess_test_hdf5
)


# ==========================================================
# Main inference function
# ==========================================================

def run_estimation(
        model_file: str | Path,
        test_file: str | Path,
        output_file: str | Path
):
    """
    Run Co-SOM inference on one challenge test catalog.

    Parameters
    ----------
    model_file : str or Path
        Trained C-4 Co-SOM model.

    test_file : str or Path
        Test HDF5 catalog.

    output_file : str or Path
        Output qp HDF5 file.
    """

    # ======================================================
    # Load trained model
    # ======================================================

    print("\nLoading model...")

    model = load_submission_model(
        model_file
    )

    # ======================================================
    # Preprocess test data
    # ======================================================

    print("\nProcessing test catalog...")

    X_test, object_id = preprocess_test_hdf5(
        test_file,
        model["metadata"]["preprocess_params"]
    )

    print(
        "Objects:",
        X_test.shape[0]
    )

    # ======================================================
    # BMU search
    # ======================================================

    print("\nSearching BMUs...")

    matches = find_best_bmu(
        X_test,
        model["regions"],
        k_candidates=10
    )

    # ======================================================
    # Predict p(z) - C4
    # ======================================================

    print("\nPredicting PDFs...")

    predictions = predict_objects(
        matches,
        model
    )

    # ======================================================
    # Save qp
    # ======================================================

    print("\nSaving qp file...")

    save_qp_submission(
        pdf=predictions["pdf"],
        z_edges=model["metadata"]["z_edges_global"],
        object_id=object_id,
        filename=output_file
    )

    print("\nInference finished.")


# ==========================================================
# Command line execution
# ==========================================================

if __name__ == "__main__":

    MODEL_FILE = "benchmark_C4_task4.pkl"

    TEST_FILE = "pz_challenge_taskset_4_flagship_test_1yr.hdf5"

    OUTPUT_FILE = "benchmark_C4_task4_test.hdf5"

    run_estimation(
        model_file=MODEL_FILE,
        test_file=TEST_FILE,
        output_file=OUTPUT_FILE
    )
