#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 13:35:00 2026

@author: alvaro
"""

"""
Training pipeline for Co-SOM submission.

This module trains the Co-SOM model using an HDF5 training file
and stores a reduced inference model.
"""


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

    "points_per_neuron": 20,

    "learning_rate_0": 1.0,

    "epochs": 1000,


    # Co-training

    "rounds": 10,

    "beta": 0.1,

    "m": 0.5,

    "threshold": 0.95,


    # Label fraction

    "labeled_fraction": 0.9,


    # GPU

    "batch_size": 65536,

    "mini_batch_size": 2048

}



# ==========================================================
# Main training function
# ==========================================================


def train_submission(
        train_file,
        output_model
):
    """
    Train Co-SOM model for a challenge training file.

    Parameters
    ----------
    train_file : str or Path
        Input HDF5 training catalog.

    output_model : str or Path
        Output pickle model file.
    """



    print(
        "\nLoading training data..."
    )


    (
        X_train,
        z_train,
        object_id,
        preprocess_params

    ) = preprocess_train_hdf5(
        train_file
    )



    print(
        "Objects:",
        len(z_train)
    )


    print(
        "Features:",
        X_train.shape[1]
    )



    # ------------------------------------------------------
    # Global redshift grid
    # ------------------------------------------------------

    params["z_edges_global"] = np.arange(

        z_train.min(),

        z_train.max() + 0.001,

        0.001

    )



    # ------------------------------------------------------
    # Create training regions
    # ------------------------------------------------------

    print(
        "\nCreating regions..."
    )


    regions = create_training_regions(

        X_train,

        z_train,

        cuts=2,

        overlap=0.80

    )



    print(
        "Regions:",
        len(regions)
    )



    # ------------------------------------------------------
    # Train each region
    # ------------------------------------------------------

    submission_regions = []


    for region in regions:


        print(
            "\n=============================="
        )

        print(
            "Training region:",
            region["index"]
        )

        print(
            "=============================="
        )



        params["region_id"] = (
            region["index"]
        )

        params["region_lower"] = (
            region["lower"]
        )

        params["region_upper"] = (
            region["upper"]
        )



        model_region = train_region(

        region["X"],

        region["z"],

        params

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
                params["z_edges_global"]

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


    print(
        "\nSaved model:"
    )

    print(
        output_model
    )


    return output_model





# ==========================================================
# Local test
# ==========================================================


if __name__ == "__main__":


    train_submission(

        train_file=
        "pz_challenge_taskset_1_cardinal_training_1yr.hdf5",

        output_model=
        "cosom_model.pkl"

    )
