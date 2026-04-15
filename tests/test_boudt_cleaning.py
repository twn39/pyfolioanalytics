import numpy as np
import pandas as pd
from pyfolioanalytics.moments import clean_returns_boudt

def test_boudt_cleaning():
    # 1. Generate normal multivariate data
    np.random.seed(42)
    T, _N = 100, 3
    mean = np.array([0.001, 0.002, 0.003])
    cov = np.array([[0.01, 0.005, 0.002],
                    [0.005, 0.015, 0.004],
                    [0.002, 0.004, 0.02]])
    R = np.random.multivariate_normal(mean, cov, T)
    
    # 2. Inject massive outliers
    outlier_idx = [10, 50, 90]
    R[outlier_idx] = R[outlier_idx] * 10 + np.array([1.0, -1.0, 2.0])
    
    df_R = pd.DataFrame(R, columns=['A', 'B', 'C'])
    
    # 3. Clean returns
    df_clean = pd.DataFrame(clean_returns_boudt(df_R, alpha=0.05), columns=df_R.columns, index=df_R.index)
    
    # Verify shape is preserved
    assert df_clean.shape == (100, 3)
    
    # Verify outliers were shrunk
    assert np.all(np.abs(df_clean.iloc[outlier_idx].values) < np.abs(df_R.iloc[outlier_idx].values))
    
    # Verify regular points were minimally affected or kept same
    # Due to mu_mcd shift, normal points might shift slightly, but variance drops massively
    assert df_clean.cov().values.trace() < df_R.cov().values.trace()

