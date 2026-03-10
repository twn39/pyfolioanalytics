import numpy as np
import pandas as pd
from pyfolioanalytics.dbht import DBHTs
from pyfolioanalytics.ml import hrp_optimization, nco_optimization

def test_dbht_core_smoke():
    # Setup small synthetic data
    np.random.seed(42)
    T, N = 100, 10
    R = np.random.randn(T, N)
    corr = np.corrcoef(R, rowvar=False)
    dist = np.sqrt(0.5 * (1 - corr))
    S = corr + 1.0
    
    T8, Rpm, Adjv, Dpm, Mv, Z = DBHTs(dist, S, leaf_order=False)
    
    assert T8.shape == (N,)
    assert Z.shape == (N - 1, 4)
    # Check linkage format
    assert np.all(Z[:, 2] >= 0) # Distances non-negative

def test_hrp_dbht_integration():
    np.random.seed(42)
    T, N = 50, 12 # DBHT likes N >= 9
    R_raw = np.random.randn(T, N) * 0.01
    asset_names = [f"Asset.{i+1}" for i in range(N)]
    R_df = pd.DataFrame(R_raw, columns=asset_names)
    
    # Test HRP with DBHT clustering
    w_hrp = hrp_optimization(R_df, clustering="DBHT")
    
    assert isinstance(w_hrp, pd.Series)
    assert np.allclose(w_hrp.sum(), 1.0)
    assert np.all(w_hrp > 0)

def test_nco_dbht_integration():
    np.random.seed(42)
    T, N = 50, 12
    R_raw = np.random.randn(T, N) * 0.01
    asset_names = [f"Asset.{i+1}" for i in range(N)]
    R_df = pd.DataFrame(R_raw, columns=asset_names)
    
    # Test NCO with DBHT clustering
    # DBHT provides clusters (T8)
    w_nco = nco_optimization(R_df, clustering="DBHT")
    
    assert isinstance(w_nco, pd.Series)
    assert np.allclose(w_nco.sum(), 1.0)
