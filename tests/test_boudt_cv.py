import numpy as np
import pandas as pd
import pytest
import json
import os
from pyfolioanalytics.moments import clean_returns_boudt

def test_boudt_cleaning_parity():
    # Load ground truth
    json_path = "data/boudt_cv.json"
    if not os.path.exists(json_path):
        pytest.skip("data/boudt_cv.json not found. Run scripts/generate_boudt_cv.R first.")
        
    with open(json_path, "r") as f:
        gt = json.load(f)
        
    R_r = np.array(gt["original_returns"])
    R_clean_r = np.array(gt["cleaned_returns"])
    R = pd.DataFrame(R_r)
    
    # We know that Python's sklearn.covariance.MinCovDet and R's robustbase::covMcd 
    # differ internally in their exact fastMCD implementations (especially the support fraction).
    # R usually defaults to alpha=0.5 (subset size (N+p+1)/2) for covMcd.
    # Scikit-learn defaults to assume_centered=False and support_fraction=None.
    # What we care about is that BOTH algorithms detect the exact same structural outliers 
    # and shrink them appropriately.
    
    R_clean_py = clean_returns_boudt(R, alpha=0.05).values
    
    # Find outliers identified by R
    r_diff = np.abs(R_r - R_clean_r)
    r_outliers = r_diff > 1e-4
    
    # Find outliers identified by Python
    py_diff = np.abs(R_r - R_clean_py)
    py_outliers = py_diff > 1e-4
    
    # Assert that Python caught all the major outliers that R caught
    # We check if the overlap is extremely high
    overlap = np.logical_and(r_outliers, py_outliers)
    r_outlier_count = np.sum(r_outliers)
    overlap_count = np.sum(overlap)
    
    # At least 80% overlap in outlier detection is expected due to heuristic differences
    assert overlap_count / r_outlier_count > 0.80
    
    # Check that the two massive injected outliers are ALWAYS caught and shrunken
    # R.iloc[9, 0] and R.iloc[49, 2]
    assert np.abs(R_clean_py[9, 0]) < np.abs(R_r[9, 0]) * 0.5 # shrunken significantly
    assert np.abs(R_clean_py[49, 2]) < np.abs(R_r[49, 2]) * 0.5

