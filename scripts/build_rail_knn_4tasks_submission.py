#!/usr/bin/env python3
#mariano.dominguez@unc.edu.ar july 2026
"""Build precomputed rail_knn_4tasks submission tarball."""
import math
import os
import tarfile

import numpy as np
import tables_io
from rail.core.data import TableHandle
from rail.estimation.algos.k_nearneigh import KNearNeighEstimator, KNearNeighInformer
from rail.utils import catalog_utils

PUBLIC_AREA = os.environ.get("PUBLIC_AREA", "tests/public")
OUT_DIR = os.environ.get("OUT_DIR", "build_submission/rail_knn_4tasks")
TASKSETS = [1, 2, 3, 4]
SIMS = ["cardinal", "flagship"]
SCENARIOS = ["1yr", "10yr"]
_CHUNK = 150_000
_ZMAX = 3.0
_NZ = 151


def clean_training_file(train_file: str) -> str:
    data = tables_io.read(train_file)
    bad_mask = np.isnan(data["redshift"])
    if not bad_mask.any():
        return train_file
    cleaned_path = train_file.replace(".hdf5", "_cleaned.hdf5")
    cleaned_data = {key: val[~bad_mask] for key, val in data.items()}
    tables_io.write(cleaned_data, cleaned_path)
    return cleaned_path


def make_informer() -> KNearNeighInformer:
    return KNearNeighInformer.make_stage(
        name="inform",
        hdf5_groupname="",
        zmax=_ZMAX,
        nzbins=_NZ,
        chunk_size=_CHUNK,
        nondetect_val=math.nan,
        trainfrac=0.2,
        nneigh_min=3,
        nneigh_max=5,
        ngrid_sigma=6,
    )


def make_estimator(model) -> KNearNeighEstimator:
    return KNearNeighEstimator.make_stage(
        name="estimate",
        model=model,
        hdf5_groupname="",
        output_mode="return",
        nzbins=_NZ,
        zmax=_ZMAX,
        chunk_size=_CHUNK,
        nondetect_val=math.nan,
    )


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    catalog_utils.clear()
    catalog_utils.load_yaml("tests/catalogs.yaml")
    catalog_utils.apply("cardinal_roman_rubin")

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

                pz_out = make_estimator(model).estimate(test_data)
                pz_out.data.ancil["object_id"] = np.asarray(test_data()["object_id"])
                pz_out.path = estimate_path
                pz_out.write()
                print(f"Wrote {model_path} and {estimate_path}")

    tarball = "rail_knn_4tasks_submission.tgz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(OUT_DIR, arcname="rail_knn_4tasks")
    print(f"Created {tarball}")


if __name__ == "__main__":
    main()
