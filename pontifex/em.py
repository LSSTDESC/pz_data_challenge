import logging
import numpy as np
import pandas as pd
import astropy.table as at
from scipy.ndimage import gaussian_filter1d
try:
    import nugundam as ng
except ImportError:
    ng = None

logger = logging.getLogger(__name__)

def generate_footprint_randoms_combined(ref_df: pd.DataFrame, unk_df: pd.DataFrame, columns=('ra', 'dec'), order_sparse=8):
    """
    Generate footprint-corrected random catalogs using SkyKatana.
    Falls back to bounding box randoms if the catalogs are too small (e.g. in CI)
    to prevent empty array failures in isolated pixel removal.
    """
    combined = pd.concat([ref_df[['ra', 'dec']], unk_df[['ra', 'dec']]], ignore_index=True)
    
    if len(combined) < 1500:
        logger.info("Catalog size too small for sparse footprint masking. Falling back to uniform bounding box randoms.")
        rng = np.random.default_rng(42)
        ra_min, ra_max = combined['ra'].min(), combined['ra'].max()
        dec_min, dec_max = combined['dec'].min(), combined['dec'].max()
        
        ref_rands = pd.DataFrame({
            'ra': rng.uniform(ra_min, ra_max, len(ref_df) * 3),
            'dec': rng.uniform(dec_min, dec_max, len(ref_df) * 3)
        })
        unk_rands = pd.DataFrame({
            'ra': rng.uniform(ra_min, ra_max, len(unk_df) * 3),
            'dec': rng.uniform(dec_min, dec_max, len(unk_df) * 3)
        })
        return ref_rands, unk_rands
        
    try:
        from skykatana import SkyMaskPipe
        logger.info("Building SkyKatana footprint mask on combined sample...")
        pipe = SkyMaskPipe(order_out=10)
        pipe.build_foot_mask(combined, columns=columns, order_sparse=order_sparse, remove_isopixels=False, erode_borders=False)
        
        logger.info("Generating footprint-corrected random catalogs...")
        ref_rands = pipe.makerans(stage='footmask', nr=len(ref_df) * 3)
        unk_rands = pipe.makerans(stage='footmask', nr=len(unk_df) * 3)
        return ref_rands, unk_rands
    except Exception as e:
        logger.warning(f"SkyKatana footprint masking failed with: {e}. Falling back to uniform bounding box randoms.")
        rng = np.random.default_rng(42)
        ra_min, ra_max = combined['ra'].min(), combined['ra'].max()
        dec_min, dec_max = combined['dec'].min(), combined['dec'].max()
        
        ref_rands = pd.DataFrame({
            'ra': rng.uniform(ra_min, ra_max, len(ref_df) * 3),
            'dec': rng.uniform(dec_min, dec_max, len(ref_df) * 3)
        })
        unk_rands = pd.DataFrame({
            'ra': rng.uniform(ra_min, ra_max, len(unk_df) * 3),
            'dec': rng.uniform(dec_min, dec_max, len(unk_df) * 3)
        })
        return ref_rands, unk_rands


class PontifexEM:
    """
    Expectation-Maximization (EM) optimization of photometric redshift PDFs
    using Nugundam clustering cross-correlations and SkyKatana masks.
    """
    def __init__(
        self,
        unk_df,
        ref_df,
        initial_pdfs,
        z_grid_edges,
        cosmo=None,
        pimax=100.0,
        unk_rands_df=None,
        ref_rands_df=None,
        mc_nreal=5,
        nthreads=4,
        use_bias_correction=True,
        floor_val=1e-5,
        learning_rate=0.2,
        smoothing_sigma=1.0,
        is_ci=False
    ):
        self.unk_df = unk_df.copy()
        self.ref_df = ref_df.copy()
        
        # Sanitize spatial coordinates in unk_df and ref_df
        for df in [self.unk_df, self.ref_df]:
            for col in ['ra', 'dec']:
                if col in df.columns:
                    vals = np.asarray(df[col], dtype=float)
                    nan_inf = np.isnan(vals) | np.isinf(vals)
                    if np.sum(nan_inf) > 0:
                        med_val = float(np.nanmedian(vals)) if not np.isnan(np.nanmedian(vals)) else 0.0
                        vals[nan_inf] = med_val
                        df[col] = vals
                        logger.warning(f"PontifexEM Guard: Fixed {np.sum(nan_inf)} invalid {col} entries using median={med_val:.4f}.")

        # Sanitize initial PDFs
        pdfs_arr = np.array(initial_pdfs, dtype=float)
        pdfs_arr = np.nan_to_num(pdfs_arr, nan=0.0, posinf=0.0, neginf=0.0)
        pdfs_arr = np.maximum(pdfs_arr, 0.0)
        row_sums = pdfs_arr.sum(axis=1, keepdims=True)
        self.initial_pdfs = np.where(row_sums > 0, pdfs_arr / row_sums, 1.0 / pdfs_arr.shape[1])
        
        self.z_grid_edges = np.array(z_grid_edges, dtype=float)
        self.z_bin_centers = 0.5 * (self.z_grid_edges[:-1] + self.z_grid_edges[1:])
        
        if cosmo is None:
            # Default to LSST DESC standard cosmology
            if ng is not None:
                self.cosmo = ng.DistanceSpec(calcdist=True, h0=67.27, omegam=0.3121, omegal=0.6879)
            else:
                self.cosmo = None
        else:
            self.cosmo = cosmo
            
        self.pimax = pimax
        self.mc_nreal = 1 if is_ci else mc_nreal
        self.nthreads = 1 if is_ci else nthreads
        self.use_bias_correction = use_bias_correction
        self.floor_val = floor_val
        self.learning_rate = learning_rate
        self.smoothing_sigma = smoothing_sigma
        self.is_ci = is_ci
        
        # Verify sizes
        if len(self.unk_df) != self.initial_pdfs.shape[0]:
            raise ValueError(f"Rows in unk_df ({len(self.unk_df)}) != initial_pdfs rows ({self.initial_pdfs.shape[0]})")
        if len(self.z_bin_centers) != self.initial_pdfs.shape[1]:
            raise ValueError(f"Z bins ({len(self.z_bin_centers)}) != initial_pdfs cols ({self.initial_pdfs.shape[1]})")
            
        # Build randoms using SkyKatana footprint-correction if not provided
        if ref_rands_df is None or unk_rands_df is None:
            self.ref_rands_df, self.unk_rands_df = generate_footprint_randoms_combined(self.ref_df, self.unk_df)
        else:
            self.ref_rands_df = ref_rands_df.copy()
            self.unk_rands_df = unk_rands_df.copy()
            
        rng = np.random.default_rng(1234)
        if 'ztrue' not in self.ref_rands_df.columns:
            self.ref_rands_df['ztrue'] = rng.choice(self.ref_df['ztrue'], len(self.ref_rands_df))
        if 'ztrue' not in self.unk_rands_df.columns:
            self.unk_rands_df['ztrue'] = rng.choice(self.ref_df['ztrue'], len(self.unk_rands_df))
            
        self.target_nz = None
        self.optimized_pdfs = None
 
    def compute_target_nz_nugundam(self, pdfs, seppmin=0.2, dsepp=0.2, nsepp=10):
        unk = at.Table.from_pandas(self.unk_df)
        ref = at.Table.from_pandas(self.ref_df)
        unk_rand = at.Table.from_pandas(self.unk_rands_df)
        ref_rand = at.Table.from_pandas(self.ref_rands_df)
        
        if self.is_ci:
            if len(unk) > 500:
                idx_unk = np.random.default_rng(42).choice(len(unk), 500, replace=False)
                unk = unk[idx_unk]
                if pdfs is not None:
                    pdfs = pdfs[idx_unk]
                if len(unk_rand) > 1500:
                    idx_unk_rand = np.random.default_rng(42).choice(len(unk_rand), 1500, replace=False)
                    unk_rand = unk_rand[idx_unk_rand]
            if len(ref) > 500:
                idx_ref = np.random.default_rng(42).choice(len(ref), 500, replace=False)
                ref = ref[idx_ref]
                if len(ref_rand) > 1500:
                    idx_ref_rand = np.random.default_rng(42).choice(len(ref_rand), 1500, replace=False)
                    ref_rand = ref_rand[idx_ref_rand]
        
        if ng is None:
            logger.info("Nugundam not installed; using smoothed empirical reference N(z) target.")
            z_ref = np.asarray(self.ref_df['ztrue'], dtype=float)
            z_ref = z_ref[np.isfinite(z_ref)]
            counts, _ = np.histogram(z_ref, bins=self.z_grid_edges)
            n_clust = counts.astype(float) + self.floor_val
            return n_clust / np.sum(n_clust)

        n_bins = len(self.z_grid_edges) - 1
        n_clust = np.zeros(n_bins)
        
        binning = ng.ProjectedBinning.from_binsize(
            nsepp=nsepp, seppmin=seppmin, dsepp=dsepp, logsepp=True,
            nsepv=1, dsepv=self.pimax
        )
        
        cfg_cross = ng.ProjectedCrossConfig(
            estimator="LS",
            columns_data1=ng.ProjectedCatalogColumns(ra="ra", dec="dec", redshift=None),
            columns_random1=ng.ProjectedCatalogColumns(ra="ra", dec="dec", redshift="ztrue"),
            columns_data2=ng.ProjectedCatalogColumns(ra="ra", dec="dec", redshift="ztrue"),
            columns_random2=ng.ProjectedCatalogColumns(ra="ra", dec="dec", redshift="ztrue"),
            nthreads=self.nthreads,
            binning=binning,
            grid=ng.ProjectedGridSpec(autogrid=True, pxorder="natural"),
            distance=self.cosmo
        )
        cfg_cross.mc_pdf.enabled = True
        cfg_cross.mc_pdf.sample_within_bin = True
        cfg_cross.mc_pdf.nreal = self.mc_nreal
        cfg_cross.mc_pdf.z_grid = self.z_grid_edges
        cfg_cross.mc_pdf.grid_kind = 'edges'
        cfg_cross.mc_pdf.pdf_data1 = ng.PDFSourceSpec(kind="external_matrix", matrix=pdfs)
        cfg_cross.mc_pdf.store_realizations = False
        cfg_cross.mc_pdf.resampling_backend = "fast"
        cfg_cross.mc_pdf.resampling_random_policy = "fixed"
        cfg_cross.progress.enabled = False
        
        cfg_auto = ng.ProjectedAutoConfig(
            estimator="LS",
            columns_data=ng.ProjectedCatalogColumns(ra="ra", dec="dec", redshift="ztrue"),
            columns_random=ng.ProjectedCatalogColumns(ra="ra", dec="dec", redshift="ztrue"),
            nthreads=self.nthreads,
            binning=binning,
            grid=ng.ProjectedGridSpec(autogrid=True, pxorder="natural"),
            distance=self.cosmo
        )
        cfg_auto.progress.enabled = False
        
        for k in range(n_bins):
            z_low = self.z_grid_edges[k]
            z_high = self.z_grid_edges[k+1]
            
            mask_ref = (ref['ztrue'] >= z_low) & (ref['ztrue'] < z_high)
            mask_ref_rand = (ref_rand['ztrue'] >= z_low) & (ref_rand['ztrue'] < z_high)
            
            ref_bin = ref[mask_ref]
            ref_rand_bin = ref_rand[mask_ref_rand]
            
            if len(ref_bin) < 5 or len(ref_rand_bin) < 10:
                n_clust[k] = self.floor_val
                continue
                
            try:
                res_cross = ng.pccf(unk, ref_bin, cfg_cross, random1=unk_rand, random2=ref_rand_bin)
                w_cross = np.array(res_cross.wp)
                w_cross_sum = np.sum(w_cross)
                
                if self.use_bias_correction:
                    res_auto = ng.pcf(ref_bin, ref_rand_bin, cfg_auto)
                    w_auto = np.array(res_auto.wp)
                    w_auto_sum = np.sum(w_auto)
                    
                    denom = np.sqrt(np.max([w_auto_sum, 1e-5]))
                    n_clust[k] = w_cross_sum / denom
                else:
                    n_clust[k] = w_cross_sum
            except Exception as e:
                logger.warning(f"Error computing correlation in bin {k} ([{z_low:.3f}, {z_high:.3f}]): {e}")
                n_clust[k] = self.floor_val
                
        nans = np.isnan(n_clust)
        if np.any(nans):
            x = np.arange(len(n_clust))
            n_clust[nans] = np.interp(x[nans], x[~nans], n_clust[~nans])
            
        n_clust = np.maximum(n_clust, self.floor_val)
        return n_clust / np.sum(n_clust)

    def optimize(self, max_iter=5, tol=1e-5, seppmin=0.2, dsepp=0.2, nsepp=10):
        current_pdfs = self.initial_pdfs.copy()
        eps = 1e-15
        
        logger.info(f"Starting EM optimization using nugundam (max_iter={max_iter}, tol={tol:.2e})...")
        
        for i in range(max_iter):
            rng = np.random.default_rng(1234 + i)
            # Vectorized sampling of redshifts based on current PDF
            p_matrix = np.asarray(current_pdfs, dtype=np.float64)
            p_matrix = np.nan_to_num(p_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            p_matrix = np.maximum(p_matrix, 0.0)
            row_sums = p_matrix.sum(axis=1, keepdims=True)
            p_matrix = np.where(row_sums > 0, p_matrix / row_sums, 1.0 / p_matrix.shape[1])
            
            # Normalize to guarantee sum is exactly 1 for all rows
            row_sums_norm = p_matrix.sum(axis=1, keepdims=True)
            p_matrix = p_matrix / row_sums_norm
            
            cdf = np.cumsum(p_matrix, axis=1)
            r = rng.uniform(0, 1, size=(len(p_matrix), 1))
            bin_indices = np.argmax(cdf >= r, axis=1)
            z_samples = self.z_bin_centers[bin_indices]
                
            self.unk_rands_df['ztrue'] = rng.choice(z_samples, size=len(self.unk_rands_df))
            
            target_nz = self.compute_target_nz_nugundam(
                pdfs=current_pdfs,
                seppmin=seppmin,
                dsepp=dsepp,
                nsepp=nsepp
            )
            
            if self.smoothing_sigma > 0:
                target_nz = gaussian_filter1d(target_nz, sigma=self.smoothing_sigma)
                target_nz = np.maximum(target_nz, self.floor_val)
                target_nz = target_nz / np.sum(target_nz)
                
            current_nz = np.sum(current_pdfs, axis=0)
            current_nz = np.where(current_nz == 0, eps, current_nz)
            current_nz = current_nz / np.sum(current_nz)
            
            ratio = (target_nz / current_nz) ** self.learning_rate
            new_pdfs = current_pdfs * ratio
            
            new_pdfs = np.nan_to_num(new_pdfs, nan=0.0, posinf=0.0, neginf=0.0)
            new_pdfs = np.maximum(new_pdfs, 0.0)
            row_sums = np.sum(new_pdfs, axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, eps, row_sums)
            new_pdfs = new_pdfs / row_sums
            
            diff = np.sqrt(np.mean((new_pdfs - current_pdfs) ** 2))
            current_pdfs = new_pdfs
            
            logger.info(f"[Iteration {i+1}] PDF RMS difference: {diff:.2e}")
            if diff < tol:
                logger.info(f"EM converged at iteration {i+1} (diff={diff:.2e})")
                break
        else:
            logger.warning(f"EM reached max_iter={max_iter} without converging to tol={tol:.2e}")
            
        self.optimized_pdfs = current_pdfs
        return self.optimized_pdfs

