#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:33:45 2026

@author: alvaro
"""

from sklearn.ensemble import IsolationForest
import numpy as np
import pickle


def train_confidence_model(
        X_train,
        contamination="auto",
        random_state=42,
        n_estimators=200):
    """
    Train an Isolation Forest used to estimate the confidence of
    photometric redshift predictions.

    The model is trained exclusively on the photometric feature space
    used by the Co-SOM. After training, anomaly scores are computed for
    the complete training sample and sorted. These sorted scores define
    the reference distribution that will later be used to convert new
    anomaly scores into confidence percentiles.

    Parameters
    ----------
    X_train : ndarray of shape (N_samples, N_features)
        Training feature matrix.

    contamination : float or "auto", optional
        Expected proportion of anomalies. The default ("auto") lets
        Isolation Forest estimate the threshold internally.

    random_state : int, optional
        Random seed used to obtain reproducible trees.

    n_estimators : int, optional
        Number of trees in the Isolation Forest.

    Returns
    -------
    confidence_model : dict
        Dictionary containing everything required during inference.

        model : IsolationForest
            Trained Isolation Forest.

        score_reference : ndarray
            Sorted anomaly scores computed from the training set.

        score_min : float
            Minimum anomaly score observed during training.

        score_max : float
            Maximum anomaly score observed during training.

        n_features : int
            Number of photometric features expected by the model.
    """

    # ==========================================================
    # Train Isolation Forest
    # ==========================================================

    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train)

    # ==========================================================
    # Compute anomaly scores for the training sample
    # ==========================================================

    train_scores = model.score_samples(X_train)

    # ==========================================================
    # Sort scores to build the empirical score distribution
    # ==========================================================

    score_reference = np.sort(train_scores)

    # ==========================================================
    # Build confidence model
    # ==========================================================

    confidence_model = {

        "model": model,

        "score_reference": score_reference,

        "score_min": float(score_reference[0]),

        "score_max": float(score_reference[-1]),

        "n_features": X_train.shape[1]

    }

    return confidence_model

def estimate_confidence(X, confidence_model):
    """
    Estimate prediction confidence for new objects using the empirical
    score distribution of the training sample.

    Isolation Forest anomaly scores are converted into confidence
    percentiles by comparing each score against the score distribution
    observed during training.

    Confidence values range from 0 to 100:

        100 -> Object is very similar to the training distribution.

          0 -> Object is highly different from the training distribution.

    A qualitative confidence category is also assigned.

    Parameters
    ----------
    X : ndarray of shape (N_samples, N_features)
        Feature matrix.

    confidence_model : dict
        Dictionary returned by train_confidence_model().

    Returns
    -------
    confidence : ndarray
        Confidence percentage for every object.

    category : ndarray
        Confidence category for every object.

    scores : ndarray
        Raw Isolation Forest anomaly scores.
    """

    # ==========================================================
    # Extract trained model
    # ==========================================================

    model = confidence_model["model"]

    score_reference = confidence_model["score_reference"]

    # ==========================================================
    # Compute anomaly scores
    # ==========================================================

    scores = model.score_samples(X)

    # ==========================================================
    # Convert scores into empirical confidence percentiles
    # ==========================================================

    rank = np.searchsorted(
        score_reference,
        scores,
        side="right"
    )

    confidence = (
        rank / len(score_reference)
    ) * 100.0

    # ==========================================================
    # Assign confidence categories
    # ==========================================================

    category = np.empty(
        confidence.shape,
        dtype=object
    )

    category[confidence >= 95] = "Excellent"

    category[
        (confidence >= 80) &
        (confidence < 95)
    ] = "High"

    category[
        (confidence >= 60) &
        (confidence < 80)
    ] = "Moderate"

    category[
        (confidence >= 40) &
        (confidence < 60)
    ] = "Low"

    category[
        confidence < 40
    ] = "Very Low"

    return (
        confidence,
        category,
        scores
    )


def save_confidence_model(confidence_model, filename):
    """
    Save a trained confidence model to disk.

    The complete confidence model, including the trained Isolation
    Forest and the empirical score distribution, is serialized using
    pickle.

    Parameters
    ----------
    confidence_model : dict
        Dictionary returned by train_confidence_model().

    filename : str
        Output filename.

    Returns
    -------
    None
    """

    # ==========================================================
    # Save model
    # ==========================================================

    with open(filename, "wb") as f:

        pickle.dump(
            confidence_model,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

def load_confidence_model(filename):
    """
    Load a previously trained confidence model.

    Parameters
    ----------
    filename : str
        Path to the saved confidence model.

    Returns
    -------
    confidence_model : dict
        Dictionary containing the trained Isolation Forest and
        the empirical score distribution.
    """

    # ==========================================================
    # Load model
    # ==========================================================

    with open(filename, "rb") as f:

        confidence_model = pickle.load(f)

    return confidence_model