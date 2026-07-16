#!/usr/bin/env python3
"""Build precomputed tq_flexzboost submission tarball.

Trains a FlexZBoost model and writes a pre-made p(z) estimate for every
taskset x sim x scenario combination, then bundles them into a tarball whose
top-level directory matches the submission name (``tq_flexzboost``).

Requires ``tests/public/`` to be populated with the public challenge data
(e.g. via ``scripts/download_public.py`` or by copying the repo's ``public/``).
"""
import math
import os
import tarfile

import numpy as np
import tables_io
from rail.core.data import TableHandle
from rail.estimation.algos.flexzboost import FlexZBoostEstimator, FlexZBoostInformer
from rail.utils import catalog_utils

PUBLIC_AREA = os.environ.get("PUBLIC_AREA", "tests/public")
OUT_DIR = os.environ.get("OUT_DIR", "build_submission/tq_flexzboost")
CATALOG_TAG = "cardinal_roman_rubin"
TASKSETS = [1, 2, 3, 4]
SIMS = ["cardinal", "flagship"]
SCENARIOS = ["1yr", "10yr"]
_ZMIN = 0.0
_ZMAX = 6.0
_NZ = 601
_ZGRID = np.linspace(_ZMIN, _ZMAX, _NZ)
_POINT_ESTIMATES = ["zmode", "zbest"]


def clean_training_file(train_file: str) -> str:
    data = tables_io.read(train_file)
    bad_mask = np.isnan(data["redshift"])
    if not bad_mask.any():
        return train_file
    cleaned_path = train_file.replace(".hdf5", "_cleaned.hdf5")
    cleaned_data = {key: val[~bad_mask] for key, val in data.items()}
    tables_io.write(cleaned_data, cleaned_path)
    return cleaned_path


def make_informer() -> FlexZBoostInformer:
    # A single bump threshold and a single sharpening value, so the informer
    # skips the grid search and trains on that one combination.
    return FlexZBoostInformer.make_stage(
        name="inform",
        hdf5_groupname="",
        nondetect_val=math.nan,
        zmin=_ZMIN,
        zmax=_ZMAX,
        nzbins=_NZ,
        max_basis=50,
        nbump=1,
        bumpmin=0.02,
        bumpmax=0.02,
        nsharp=1,
        sharpmin=1.2,
        sharpmax=1.2,
    )


def make_estimator(model) -> FlexZBoostEstimator:
    return FlexZBoostEstimator.make_stage(
        name="estimate",
        model=model,
        hdf5_groupname="",
        output_mode="return",
        nondetect_val=math.nan,
        zmin=_ZMIN,
        zmax=_ZMAX,
        nzbins=_NZ,
        calculated_point_estimates=_POINT_ESTIMATES,
    )


def attach_ancil(estimator: FlexZBoostEstimator, pz_out, test_data) -> None:
    """Populate zmode/zbest and object_id on the output ensemble.

    FlexZBoost never calls the PointEstimationMixin (it only recognises the
    tokens mode/mean/median), so invoke the mixin explicitly to honour
    ``calculated_point_estimates``.
    """
    ensemble = estimator.calculate_point_estimates(pz_out.data, grid=_ZGRID)
    object_id = dict(object_id=np.asarray(test_data()["object_id"]))
    if ensemble.ancil is None:
        ensemble.set_ancil(object_id)
    else:
        ensemble.add_to_ancil(object_id)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    catalog_utils.clear()
    catalog_utils.load_yaml("tests/catalogs.yaml")
    catalog_utils.apply(CATALOG_TAG)

    for taskset in TASKSETS:
        for sim in SIMS:
            for scenario in SCENARIOS:
                train_file = (
                    f"{PUBLIC_AREA}/pz_challenge_taskset_{taskset}_{sim}_"
                    f"training_{scenario}.hdf5"
                )
                test_file = (
                    f"{PUBLIC_AREA}/pz_challenge_taskset_{taskset}_{sim}_"
                    f"test_{scenario}.hdf5"
                )
                model_path = (
                    f"{OUT_DIR}/pz_challenge_taskset_{taskset}_{sim}_"
                    f"pz_model_{scenario}.pkl"
                )
                estimate_path = (
                    f"{OUT_DIR}/pz_challenge_taskset_{taskset}_{sim}_"
                    f"pz_estimate_{scenario}.hdf5"
                )

                cleaned_train_file = clean_training_file(train_file)
                train_data = TableHandle("train", path=cleaned_train_file)
                test_data = TableHandle("test", path=test_file)

                model = make_informer().inform(train_data)
                model.path = model_path
                model.write()

                estimator = make_estimator(model)
                pz_out = estimator.estimate(test_data)
                attach_ancil(estimator, pz_out, test_data)
                pz_out.path = estimate_path
                pz_out.write()
                print(f"Wrote {model_path} and {estimate_path}")

    tarball = "tq_flexzboost_submission.tgz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(OUT_DIR, arcname="tq_flexzboost")
    print(f"Created {tarball}")


if __name__ == "__main__":
    main()
