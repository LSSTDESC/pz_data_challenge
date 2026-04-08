from rail.core.data import TableHandle
from rail.estimation.algos import sklearn_neurnet
from rail.utils import catalog_utils

TASKSETS = ["taskset_1", "taskset_2"]
SIMS = ["cardinal", "flagship"]
SCENARIOS = ["1yr", "10yr"]


catalog_utils.load_yaml("tests/catalogs.yaml")
catalog_utils.apply("cardinal_roman_rubin")


def train_and_estimate(
    taskset: str,
    sim: str,
    scenario: str,
) -> None:

    train_data = TableHandle(
        "train", path=f"public/pz_challenge_{taskset}_{sim}_training_{scenario}.hdf5"
    )
    test_data = TableHandle(
        "test", path=f"public/pz_challenge_{taskset}_{sim}_test_{scenario}.hdf5"
    )

    model_path = f"example/pz_challenge_{taskset}_{sim}_pz_model_{scenario}.pkl"
    output_path = f"example/pz_challenge_{taskset}_{sim}_pz_estimate_{scenario}.hdf5"

    informer = sklearn_neurnet.SklNeurNetInformer.make_stage(
        name=f"inform_{taskset}_{sim}_{scenario}",
        output_mode="return",
    )
    model = informer.inform(train_data)
    model.path = model_path
    model.write()

    estimator = sklearn_neurnet.SklNeurNetEstimator.make_stage(
        name=f"estimate_{taskset}_{sim}_{scenario}",
        model=model,
        output_mode="return",
    )
    pz_out = estimator.estimate(test_data)
    pz_out.data.ancil["object_id"] = test_data()["object_id"].astype(int)
    pz_out.path = output_path
    pz_out.write()


if __name__ == "__main__":

    for taskset in TASKSETS:
        for sim in SIMS:
            for scenario in SCENARIOS:
                train_and_estimate(taskset, sim, scenario)
