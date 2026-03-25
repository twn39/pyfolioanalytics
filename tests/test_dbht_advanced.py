import numpy as np
import pandas as pd
from pyfolioanalytics.dbht import DBHTs
from pyfolioanalytics.ml import hrp_optimization


def test_dbht_topology_properties():
    # Use data with clear cluster structure
    np.random.seed(42)
    T, N = 200, 15
    # Create 3 blocks of correlated assets
    R = np.random.randn(T, N) * 0.01
    R[:, 0:5] += 0.02  # Cluster 1
    R[:, 5:10] += -0.01  # Cluster 2

    # Induce correlation within blocks
    R[:, 0:5] += R[:, 0:1].copy() * 0.5
    R[:, 5:10] += R[:, 5:6].copy() * 0.5

    asset_names = [f"A{i}" for i in range(N)]
    R_df = pd.DataFrame(R, columns=asset_names)

    corr = R_df.corr().values
    dist = np.sqrt(0.5 * (1 - corr))
    S = corr + 1.0

    T8, Rpm, Adjv, Dpm, Mv, Z = DBHTs(dist, S, leaf_order=False)

    # 1. Verify clusters found something
    unique_clusters = np.unique(T8)
    assert len(unique_clusters) > 1

    # 2. Verify PMFG properties
    # PMFG should have 3N-6 edges for planar graphs
    # But PMFG_T2s might be slightly different in implementation
    # Adjacency matrix should be symmetric
    np.testing.assert_allclose(Rpm, Rpm.T)

    # 3. Verify HRP-DBHT Weights
    w_dbht = hrp_optimization(R_df, clustering="DBHT")
    w_link = hrp_optimization(R_df, clustering="linkage", linkage_method="single")

    # Weights should be different as clustering method differs
    assert not np.allclose(w_dbht.values, w_link.values, rtol=1e-2)
    assert np.allclose(w_dbht.sum(), 1.0)


def test_dbht_n_less_than_9():
    # DBHT typically expects N >= 9 due to PMFG logic
    np.random.seed(42)
    N = 5
    R = np.random.randn(50, N)
    R_df = pd.DataFrame(R)

    # Implementation should handle small N gracefully (e.g. fallback or print warning)
    # Our current implementation prints a warning in PMFG_T2s (inherited from Riskfolio)
    # Let's ensure it doesn't crash.
    w = hrp_optimization(R_df, clustering="DBHT")
    assert len(w) == N
    assert np.allclose(w.sum(), 1.0)
