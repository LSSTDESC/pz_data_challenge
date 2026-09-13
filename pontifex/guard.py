"""
Pontifex Input Protection Guard & Feature Sanitizer Module
-----------------------------------------------------------
Protects downstream expert estimators by identifying, reporting, and sanitizing
invalid values (NaN, Inf, -Inf), unphysical errors, and extreme numerical outliers
in catalog features while emitting detailed diagnostic warnings.
"""

import warnings
import logging
from typing import Dict, Any, Tuple
import numpy as np

logger = logging.getLogger("pontifex.guard")

# Default physical limits for photometric bands and coordinates
PHYSICAL_LIMITS = {
    "mag_min": 10.0,
    "mag_max": 38.0,
    "err_min": 1e-5,
    "err_max": 5.0,
    "ra_min": -360.0,
    "ra_max": 360.0,
    "dec_min": -90.0,
    "dec_max": 90.0,
    "z_min": 0.0,
    "z_max": 12.0
}


def sanitize_input_catalog(
    catalog_dict: Dict[str, Any],
    nondetect_val: float = 99.0,
    raise_warnings: bool = True
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Sanitizes an input catalog dictionary for downstream expert model consumption.
    
    Checks every feature column for:
    - NaN, Inf, -Inf values
    - Negative or zero measurement errors (err <= 0)
    - Unphysical magnitudes (< 10 or > 38)
    - Extreme numerical outliers (> 10 IQR from median)
    
    Replaces corrupted/invalid values with median/default values and emits explicit
    UserWarning diagnostics with precise counts and frequencies.
    
    Parameters
    ----------
    catalog_dict : Dict[str, Any]
        Dictionary containing catalog features (magnitudes, errors, coordinates, etc.)
    nondetect_val : float, optional
        Replacement magnitude for invalid/missing non-detections, default 99.0
    raise_warnings : bool, optional
        Whether to emit UserWarning diagnostics, default True
        
    Returns
    -------
    sanitized_dict : Dict[str, np.ndarray]
        Cleaned, robust catalog dictionary safe for all experts.
    guard_report : Dict[str, Any]
        Summary report detailing detected issues and frequency metrics.
    """
    sanitized = {}
    guard_report = {"total_records": 0, "issues_by_column": {}}
    
    if not catalog_dict:
        raise ValueError("Input catalog_dict is empty!")
        
    # Determine number of records
    first_key = list(catalog_dict.keys())[0]
    n_records = len(catalog_dict[first_key])
    guard_report["total_records"] = n_records
    
    if n_records == 0:
        raise ValueError("Input catalog dictionary contains zero records!")

    warnings_summary = []

    for col_name, raw_data in catalog_dict.items():
        arr = np.asarray(raw_data).copy()
        
        # Convert non-numeric or scalar object arrays if necessary
        if not np.issubdtype(arr.dtype, np.number) and not np.issubdtype(arr.dtype, np.bool_):
            try:
                arr = arr.astype(np.float64)
            except Exception:
                sanitized[col_name] = arr
                continue
                
        # Skip boolean or object ID columns from feature limits
        if col_name in ["object_id", "id", "ID", "objectId"] or np.issubdtype(arr.dtype, np.bool_) or np.issubdtype(arr.dtype, np.integer):
            sanitized[col_name] = arr
            continue

        arr = arr.astype(np.float64)
        col_issues = {
            "nan_count": 0,
            "inf_count": 0,
            "unphysical_err_count": 0,
            "unphysical_mag_count": 0,
            "outlier_count": 0
        }

        # 1. Detect NaN, Inf, -Inf
        nan_mask = np.isnan(arr)
        inf_mask = np.isinf(arr)
        col_issues["nan_count"] = int(np.sum(nan_mask))
        col_issues["inf_count"] = int(np.sum(inf_mask))
        
        invalid_mask = nan_mask | inf_mask

        # 2. Check for Error Column Unphysical Values (err <= 0)
        if col_name.endswith("_err") or "error" in col_name.lower() or "err" in col_name.lower():
            unphys_err = (arr <= 0) & (~invalid_mask)
            col_issues["unphysical_err_count"] = int(np.sum(unphys_err))
            invalid_mask |= unphys_err

        # 3. Check for Magnitude Unphysical Values (mag < 10 or mag > 38)
        elif col_name.startswith("mag_") or col_name.startswith("mag"):
            unphys_mag = ((arr < PHYSICAL_LIMITS["mag_min"]) | (arr > PHYSICAL_LIMITS["mag_max"])) & (~invalid_mask) & (arr != nondetect_val)
            col_issues["unphysical_mag_count"] = int(np.sum(unphys_mag))
            invalid_mask |= unphys_mag

        # 4. Detect extreme numerical outliers using IQR (> 10 * IQR from median)
        valid_vals = arr[~invalid_mask]
        if len(valid_vals) > 10:
            med = np.median(valid_vals)
            q25, q75 = np.percentile(valid_vals, [25, 75])
            iqr = q75 - q25
            if iqr > 1e-6:
                extreme_outliers = (np.abs(arr - med) > 10.0 * iqr) & (~invalid_mask)
                col_issues["outlier_count"] = int(np.sum(extreme_outliers))
                invalid_mask |= extreme_outliers

        # Calculate total corrupted entries for this column
        total_corrupted = np.sum(invalid_mask)
        if total_corrupted > 0:
            freq_pct = (total_corrupted / n_records) * 100.0
            
            # Compute replacement median value from valid entries
            if len(valid_vals) > 0:
                fill_val = float(np.median(valid_vals))
            else:
                fill_val = nondetect_val if col_name.startswith("mag") else 0.1
                
            arr[invalid_mask] = fill_val
            
            msg = (
                f"Feature Guard Alert: Column '{col_name}' contains {total_corrupted} / {n_records} "
                f"corrupted/invalid records ({freq_pct:.2f}% frequency). "
                f"Details: NaNs={col_issues['nan_count']}, Infs={col_issues['inf_count']}, "
                f"Unphysical Errs={col_issues['unphysical_err_count']}, Unphysical Mags={col_issues['unphysical_mag_count']}, "
                f"Outliers={col_issues['outlier_count']}. Imputed with fill_val={fill_val:.4f}."
            )
            warnings_summary.append(msg)
            logger.warning(msg)

        guard_report["issues_by_column"][col_name] = col_issues
        sanitized[col_name] = arr

    if raise_warnings and len(warnings_summary) > 0:
        full_warning = (
            f"\n[Pontifex Feature Guard Protection Triggered]\n"
            f"Detected {len(warnings_summary)} feature columns with corrupted or invalid records:\n"
            + "\n".join(f" - {w}" for w in warnings_summary)
        )
        warnings.warn(full_warning, UserWarning, stacklevel=2)

    return sanitized, guard_report
