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

    m = df[MAGNITUDE_COLUMNS].to_numpy(dtype=np.float32)
    e = df[MAGNITUDE_ERROR_COLUMNS].to_numpy(dtype=np.float32)

    X = np.empty((len(df), 14), dtype=np.float32)

    # -------------------------
    # Colors
    # -------------------------
    X[:, 0] = m[:, 0] - m[:, 1]
    X[:, 1] = m[:, 1] - m[:, 2]
    X[:, 2] = m[:, 2] - m[:, 3]
    X[:, 3] = m[:, 3] - m[:, 4]
    X[:, 4] = m[:, 4] - m[:, 5]
    X[:, 5] = m[:, 6] - m[:, 7]
    X[:, 6] = m[:, 7] - m[:, 8]
    
    # -------------------------
    # Color errors
    # -------------------------
    X[:, 7]  = np.sqrt(e[:, 0]**2 + e[:, 1]**2)
    X[:, 8]  = np.sqrt(e[:, 1]**2 + e[:, 2]**2)
    X[:, 9]  = np.sqrt(e[:, 2]**2 + e[:, 3]**2)
    X[:, 10] = np.sqrt(e[:, 3]**2 + e[:, 4]**2)
    X[:, 11] = np.sqrt(e[:, 4]**2 + e[:, 5]**2)
    X[:, 12] = np.sqrt(e[:, 6]**2 + e[:, 7]**2)
    X[:, 13] = np.sqrt(e[:, 7]**2 + e[:, 8]**2)

    return X


# ============================================================
# Read HDF5 challenge files
# ============================================================

def read_hdf5_catalog(filename, columns=None):
    """
    Read selected columns from an HDF5 catalog.
    """

    with h5py.File(filename, "r") as f:

        if columns is None:
            columns = list(f.keys())

        data = {
            key: f[key][:]
            for key in columns
            if key in f
        }

    return pd.DataFrame(data)
# ============================================================
# Training preprocessing parquet and HDF5
# ============================================================

def preprocess_train(train_file):

    columns = (
    ["object_id", "redshift"]
    + MAGNITUDE_COLUMNS
    + MAGNITUDE_ERROR_COLUMNS
    )

    train = pd.read_parquet(
    train_file,
    columns=columns
    )

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

def preprocess_train_hdf5(train_file, return_unlabeled=True):

    columns = (
        [
            "object_id",
            "redshift",
            "redshift_manyband",
            "COSMOS"
        ]
        + MAGNITUDE_COLUMNS
        + MAGNITUDE_ERROR_COLUMNS
    )

    train = read_hdf5_catalog(train_file, columns)

    # ---------------------------------------------------------
    # 1. Separate labeled and unlabeled objects
    #
    # Labeled:
    #   valid spectroscopic redshift
    #
    # Unlabeled:
    #   no spectroscopic redshift
    #   valid COSMOS many-band redshift
    #
    # redshift_manyband is NOT used as the training label.
    # It is preserved as auxiliary information.
    # ---------------------------------------------------------

    z_spec = train["redshift"].to_numpy(dtype=np.float32)
    z_manyband = train["redshift_manyband"].to_numpy(dtype=np.float32)
    cosmos = train["COSMOS"].to_numpy()

    labeled_mask = np.isfinite(z_spec)

    unlabeled_mask = (
        ~labeled_mask
        & np.isfinite(z_manyband)
        & cosmos.astype(bool)
    )

    print("\n===== TASK SET 3 LABEL STRUCTURE =====")
    print("Objects total:", len(train))
    print("Labeled (spectroscopic):", labeled_mask.sum())
    print("Unlabeled (COSMOS only):", unlabeled_mask.sum())
    print("Other objects:", (~labeled_mask & ~unlabeled_mask).sum())
    print("=======================================\n")

    labeled = train.loc[labeled_mask].reset_index(drop=True)
    unlabeled = train.loc[unlabeled_mask].reset_index(drop=True)

    object_id_labeled = labeled["object_id"].to_numpy()
    object_id_unlabeled = unlabeled["object_id"].to_numpy()

    # ---------------------------------------------------------
    # 2. Fit preprocessing using the labeled set
    # ---------------------------------------------------------

    knn_imputer = KNNImputer(
        n_neighbors=10,
        weights="distance"
    )

    labeled[MAGNITUDE_COLUMNS] = knn_imputer.fit_transform(
        labeled[MAGNITUDE_COLUMNS]
    )

    if len(unlabeled) > 0:
        unlabeled[MAGNITUDE_COLUMNS] = knn_imputer.transform(
            unlabeled[MAGNITUDE_COLUMNS]
        )

    # ---------------------------------------------------------
    # 3. Error imputation
    # ---------------------------------------------------------

    error_medians = {}

    for col in MAGNITUDE_ERROR_COLUMNS:

        median = labeled[col].median()
        error_medians[col] = median

        labeled[col] = labeled[col].fillna(median)

        if len(unlabeled) > 0:
            unlabeled[col] = unlabeled[col].fillna(median)

    # ---------------------------------------------------------
    # 4. Features
    # ---------------------------------------------------------

    X_labeled = create_features(labeled)
    z_labeled = labeled["redshift"].to_numpy(dtype=np.float32)

    if len(unlabeled) > 0:
        X_unlabeled = create_features(unlabeled)
        z_manyband_unlabeled = unlabeled[
            "redshift_manyband"
        ].to_numpy(dtype=np.float32)
    else:
        X_unlabeled = np.empty(
            (0, len(FEATURES)),
            dtype=np.float32
        )
        z_manyband_unlabeled = np.empty(
            (0,),
            dtype=np.float32
        )

    assert not np.isnan(X_labeled).any()
    assert not np.isnan(X_unlabeled).any()

    preprocess_params = {
        "knn_imputer": knn_imputer,
        "error_medians": error_medians
    }

    if not return_unlabeled:
        return (
            X_labeled,
            z_labeled,
            object_id_labeled,
            preprocess_params
        )

    return (
        X_labeled,
        z_labeled,
        object_id_labeled,
        preprocess_params,
        X_unlabeled,
        z_manyband_unlabeled,
        object_id_unlabeled
    )


# ============================================================
# Test preprocessing parquet and HDF5
# ============================================================

def preprocess_test(
    test_file,
    preprocess_params
):

    columns = (
        ["object_id"]
        + MAGNITUDE_COLUMNS
        + MAGNITUDE_ERROR_COLUMNS
    )

    test = pd.read_parquet(
        test_file,
        columns=columns
    )

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

    X_test = create_features(
        test
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