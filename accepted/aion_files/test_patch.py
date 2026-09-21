import os
import sys
import pytest
import joblib
import numpy as np

# Make the repository root importable to avoid import errors
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def test_patch_models():
    import rail_aion_pz
    from rail.core.data import TableHandle
    from rail.estimation.algos.pzflow_nf import PZFlowInformer
    from rail.estimation.algos.gpz import GPzInformer
    import aion_pz

    submission_name = os.environ.get("PZDC_SUBMISSION_NAME", "graysmoke")
    submit_dir = os.path.join(REPO_ROOT, f"submissions/{submission_name}")
    public_area = os.path.join(REPO_ROOT, "tests/public")
    sims = ["cardinal", "flagship"]
    scenarios = ["1yr", "10yr"]
    tasksets = [1, 2, 3, 4]

    for taskset in tasksets:
        for sim in sims:
            for scenario in scenarios:
                model_file = os.path.join(
                    submit_dir, f"pz_challenge_taskset_{taskset}_{sim}_pz_model_{scenario}.pkl"
                )
                if not os.path.exists(model_file):
                    print(f"File not found: {model_file}")
                    continue
                
                print(f"\n==================================================")
                print(f"Loading {model_file}...")
                model_dict = joblib.load(model_file)
                
                # Check if pzflow or gpz are missing
                if "model_pzflow_bytes" not in model_dict or "model_gpz" not in model_dict:
                    print(f"Patching {model_file}...")
                    train_file = os.path.join(
                        public_area, f"pz_challenge_taskset_{taskset}_{sim}_training_{scenario}.hdf5"
                    )
                    train_dict = aion_pz.load_catalog(train_file)
                    
                    # Subsample if missing redshifts
                    z_true = np.asarray(train_dict[aion_pz.REDSHIFT_COL], dtype="float64")
                    if aion_pz.MANYBAND_COL in train_dict:
                        z_many = np.asarray(train_dict[aion_pz.MANYBAND_COL], dtype="float64")
                        fill = ~np.isfinite(z_true) & np.isfinite(z_many)
                        z_true[fill] = z_many[fill]
                    good = np.isfinite(z_true)
                    train_dict = {k: v[good] for k, v in train_dict.items()}
                    
                    # Subsample to keep patching fast and robust
                    n_train = len(train_dict[list(train_dict.keys())[0]])
                    max_patch_train = 500
                    if n_train > max_patch_train:
                        idx = np.sort(np.random.default_rng(0).choice(n_train, max_patch_train, replace=False))
                        train_dict = {k: v[idx] for k, v in train_dict.items()}
                        print(f"Subsampled training set to {max_patch_train} objects for patching.")
                    
                    train_handle = TableHandle('train_data', data=train_dict)
                    
                    bands = model_dict["bands"]
                    ref_band = model_dict["ref_band"]
                    err_bands = model_dict["err_bands"]
                    
                    # Mag limits
                    full_mag_limits = {
                        'mag_u_lsst': 26.4,
                        'mag_g_lsst': 27.8,
                        'mag_r_lsst': 27.1,
                        'mag_i_lsst': 26.7,
                        'mag_z_lsst': 25.8,
                        'mag_y_lsst': 24.6,
                        'mag_Y_roman': 26.5,
                        'mag_J_roman': 26.5,
                        'mag_H_roman': 26.5
                    }
                    mag_limits = {k: v for k, v in full_mag_limits.items() if k in bands}
                    
                    # Clean temporary pzflow model file if exists
                    pzflow_tmp_path = "pzflow_model.pkl"
                    if os.path.exists(pzflow_tmp_path):
                        os.remove(pzflow_tmp_path)
                        
                    # Train PZFlow
                    print("Training PZFlow...")
                    pzflow_inf = rail_aion_pz.make_clean_stage(
                        PZFlowInformer,
                        name="inform_pzflow", model=pzflow_tmp_path, hdf5_groupname="",
                        zmin=0.03, zmax=3.0, nzbins=300, seed=0,
                        ref_band=ref_band, column_names=bands, mag_limits=mag_limits,
                        include_mag_errors=False, redshift_col="redshift",
                        n_training_epochs=2
                    )
                    pzflow_model = pzflow_inf.inform(train_handle)
                    
                    # Read the bytes of the saved PZFlow model file to bypass pickling error
                    if os.path.exists(pzflow_tmp_path):
                        with open(pzflow_tmp_path, "rb") as f:
                            model_dict["model_pzflow_bytes"] = f.read()
                        os.remove(pzflow_tmp_path)
                        print("Serialized PZFlow model as bytes.")
                    else:
                        raise FileNotFoundError(f"PZFlow model file {pzflow_tmp_path} not found after training!")
                    
                    # Set model_pzflow to None (avoiding pickling error)
                    model_dict["model_pzflow"] = None
                    
                    # Train GPz
                    print("Training GPz...")
                    gpz_inf = rail_aion_pz.make_clean_stage(
                        GPzInformer,
                        name="inform_gpz", model="gpz_model.pkl", hdf5_groupname="",
                        bands=bands, err_bands=err_bands, ref_band=ref_band, redshift_col="redshift",
                        replace_error_vals=[0.1] * len(bands), max_iter=5, n_basis=10,
                        train_frac=0.8, csl_method="normal", mag_limits=mag_limits
                    )
                    gpz_model = gpz_inf.inform(train_handle)
                    model_dict["model_gpz"] = gpz_model
                    
                    # Save back
                    joblib.dump(model_dict, model_file, compress=3)
                    print(f"Patched {model_file} successfully!")
                else:
                    print(f"Model already contains PZFlow and GPz models.")
