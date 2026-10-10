import numpy as np
import pandas as pd
import h5py
from sklearn.impute import KNNImputer


# ============================================================
# Columns
# ============================================================

MAGNITUDE_COLUMNS = [
    "mag_u_lsst",
    "mag_g_lsst",
    "mag_r_lsst",
    "mag_i_lsst",
    "mag_z_lsst",
    "mag_y_lsst",
    "mag_Y_roman",
    "mag_J_roman",
    "mag_H_roman",
]

MAGNITUDE_ERROR_COLUMNS = [
    "mag_u_lsst_err",
    "mag_g_lsst_err",
    "mag_r_lsst_err",
    "mag_i_lsst_err",
    "mag_z_lsst_err",
    "mag_y_lsst_err",
    "mag_Y_roman_err",
    "mag_J_roman_err",
    "mag_H_roman_err",
]

COLOR_FEATURES = [
    "u_g",
    "g_r",
    "r_i",
    "i_z",
    "z_y",
    "Y_J",
    "J_H",
]

ERROR_FEATURES = [
    "u_g_err",
    "g_r_err",
    "r_i_err",
    "i_z_err",
    "z_y_err",
    "Y_J_err",
    "J_H_err",
]

# Features used by the SOM
FEATURES = (
    COLOR_FEATURES +
    ERROR_FEATURES
)


# ============================================================
# Create colors and propagated color errors
# ============================================================

def create_features(df):

    df = df.copy()

    # -------------------------
    # Colors
    # -------------------------

    df["u_g"] = df["mag_u_lsst"] - df["mag_g_lsst"]
    df["g_r"] = df["mag_g_lsst"] - df["mag_r_lsst"]
    df["r_i"] = df["mag_r_lsst"] - df["mag_i_lsst"]
    df["i_z"] = df["mag_i_lsst"] - df["mag_z_lsst"]
    df["z_y"] = df["mag_z_lsst"] - df["mag_y_lsst"]

    df["Y_J"] = df["mag_Y_roman"] - df["mag_J_roman"]
    df["J_H"] = df["mag_J_roman"] - df["mag_H_roman"]

    # -------------------------
    # Color errors
    # -------------------------

    df["u_g_err"] = np.sqrt(
        df["mag_u_lsst_err"]**2 +
        df["mag_g_lsst_err"]**2
    )

    df["g_r_err"] = np.sqrt(
        df["mag_g_lsst_err"]**2 +
        df["mag_r_lsst_err"]**2
    )

    df["r_i_err"] = np.sqrt(
        df["mag_r_lsst_err"]**2 +
        df["mag_i_lsst_err"]**2
    )

    df["i_z_err"] = np.sqrt(
        df["mag_i_lsst_err"]**2 +
        df["mag_z_lsst_err"]**2
    )

    df["z_y_err"] = np.sqrt(
        df["mag_z_lsst_err"]**2 +
        df["mag_y_lsst_err"]**2
    )

    df["Y_J_err"] = np.sqrt(
        df["mag_Y_roman_err"]**2 +
        df["mag_J_roman_err"]**2
    )

    df["J_H_err"] = np.sqrt(
        df["mag_J_roman_err"]**2 +
        df["mag_H_roman_err"]**2
    )

    return df

# ============================================================
# Read HDF5 challenge files
# ============================================================

def read_hdf5_catalog(filename):
    """
    Read HDF5 files and convert them into a pandas DataFrame.
    """

    with h5py.File(
        filename,
        "r"
    ) as f:


        data = {}


        for key in f.keys():

            data[key] = f[key][:]


    df = pd.DataFrame(data)


    return df

# ============================================================
# Training preprocessing parquet and HDF5
# ============================================================

def preprocess_train(train_file):

    train = pd.read_parquet(train_file)

    object_id = train["object_id"].to_numpy()

    # ---------------------------------------------------------
    # 1. Impute magnitudes using KNN
    # ---------------------------------------------------------

    knn_imputer = KNNImputer(
        n_neighbors=10,
        weights="distance"
    )

    train[MAGNITUDE_COLUMNS] = knn_imputer.fit_transform(
        train[MAGNITUDE_COLUMNS]
    )

    # ---------------------------------------------------------
    # 2. Impute magnitude errors using training medians
    # ---------------------------------------------------------

    error_medians = {}

    for col in MAGNITUDE_ERROR_COLUMNS:

        median = train[col].median()

        error_medians[col] = median

        train[col] = train[col].fillna(median)

    # ---------------------------------------------------------
    # 3. Create colors
    # ---------------------------------------------------------

    train = create_features(train)

    # ---------------------------------------------------------
    # 4. Feature matrix
    # ---------------------------------------------------------

    X_train = train[FEATURES].to_numpy(dtype=np.float32)

    z_train = train["redshift"].to_numpy(dtype=np.float32)

    assert not np.isnan(X_train).any()

    preprocess_params = {

        "knn_imputer": knn_imputer,

        "error_medians": error_medians

    }

    return (
        X_train,
        z_train,
        object_id,
        preprocess_params
    )

def preprocess_train_hdf5(train_file):


    train = read_hdf5_catalog(
        train_file
    )


    object_id = train["object_id"].to_numpy()



    # ---------------------------------------------------------
    # Impute magnitudes
    # ---------------------------------------------------------

    knn_imputer = KNNImputer(
        n_neighbors=10,
        weights="distance"
    )


    train[MAGNITUDE_COLUMNS] = knn_imputer.fit_transform(
        train[MAGNITUDE_COLUMNS]
    )



    # ---------------------------------------------------------
    # Error imputation
    # ---------------------------------------------------------

    error_medians = {}


    for col in MAGNITUDE_ERROR_COLUMNS:

        median = train[col].median()

        error_medians[col] = median

        train[col] = train[col].fillna(
            median
        )



    # ---------------------------------------------------------
    # Features
    # ---------------------------------------------------------

    train = create_features(
        train
    )



    X_train = train[
        FEATURES
    ].to_numpy(
        dtype=np.float32
    )



    z_train = train[
        "redshift"
    ].to_numpy(
        dtype=np.float32
    )



    preprocess_params = {

        "knn_imputer":
            knn_imputer,

        "error_medians":
            error_medians

    }


    return (

        X_train,

        z_train,

        object_id,

        preprocess_params

    )

# ============================================================
# Test preprocessing parquet and HDF5
# ============================================================

def preprocess_test(
    test_file,
    preprocess_params
):

    test = pd.read_parquet(test_file)

    object_id = test["object_id"].to_numpy()

    # ---------------------------------------------------------
    # 1. Impute magnitudes
    # ---------------------------------------------------------

    test[MAGNITUDE_COLUMNS] = preprocess_params[
        "knn_imputer"
    ].transform(
        test[MAGNITUDE_COLUMNS]
    )

    # ---------------------------------------------------------
    # 2. Impute magnitude errors
    # ---------------------------------------------------------

    for col in MAGNITUDE_ERROR_COLUMNS:

        test[col] = test[col].fillna(
            preprocess_params["error_medians"][col]
        )

    # ---------------------------------------------------------
    # 3. Create colors
    # ---------------------------------------------------------

    test = create_features(test)

    # ---------------------------------------------------------
    # 4. Feature matrix
    # ---------------------------------------------------------

    X_test = test[FEATURES].to_numpy(dtype=np.float32)

    assert not np.isnan(X_test).any()

    return (
        X_test,
        object_id
    )


def preprocess_test_hdf5(
        test_file,
        preprocess_params
):


    test = read_hdf5_catalog(
        test_file
    )


    object_id = test[
        "object_id"
    ].to_numpy()



    # ---------------------------------------------------------
    # Magnitudes
    # ---------------------------------------------------------

    test[MAGNITUDE_COLUMNS] = preprocess_params[
        "knn_imputer"
    ].transform(
        test[MAGNITUDE_COLUMNS]
    )



    # ---------------------------------------------------------
    # Errors
    # ---------------------------------------------------------

    for col in MAGNITUDE_ERROR_COLUMNS:

        test[col] = test[col].fillna(

            preprocess_params[
                "error_medians"
            ][col]

        )



    # ---------------------------------------------------------
    # Features
    # ---------------------------------------------------------

    test = create_features(
        test
    )



    X_test = test[
        FEATURES
    ].to_numpy(
        dtype=np.float32
    )



    assert not np.isnan(
        X_test
    ).any()



    return (

        X_test,

        object_id

    )

def create_training_regions(
    X,
    z,
    cuts=2,
    overlap=0.80
):
    """
    Split the training set into redshift regions with optional overlap.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (N_samples, N_features).

    z : np.ndarray
        Spectroscopic redshift labels.

    cuts : int, optional
        Number of redshift regions.

    overlap : float, optional
        Overlap factor between consecutive regions.

    Returns
    -------
    regions : list of dict

        Each dictionary contains:

        {
            "X"      : feature matrix,
            "z"      : redshifts,
            "lower"  : lower redshift limit,
            "upper"  : upper redshift limit,
            "index"  : region number
        }
    """

    z = np.asarray(z)
    X = np.asarray(X)

    z_min = z.min()
    z_max = z.max()

    edges = np.linspace(
        z_min,
        z_max,
        cuts + 1
    )

    regions = []

    for i in range(cuts):

        lower = edges[i]
        upper = edges[i + 1]

        # Apply overlap except for the first region
        if i > 0:
            lower *= overlap

        # Last region extends to the maximum redshift
        if i == cuts - 1:
            mask = z >= lower
            upper = z_max

        else:
            mask = (
                (z >= lower) &
                (z < upper)
            )

        regions.append({

            "index": i,

            "lower": lower,

            "upper": upper,

            "X": X[mask],

            "z": z[mask]

        })

        print(
            f"Region {i+1}/{cuts}: "
            f"{mask.sum()} objects "
            f"[{lower:.4f}, {upper:.4f}]"
        )

    return regions