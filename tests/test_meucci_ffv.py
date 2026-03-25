import numpy as np
import pandas as pd
from pyfolioanalytics.meucci import (
    meucci_views,
    meucci_moments,
    meucci_ranking,
    entropy_prog,
)


def test_meucci_relative_view():
    np.random.seed(42)
    T, N = 100, 2
    # Create two assets where B has higher mean than A
    R_raw = np.random.randn(T, N) * 0.01
    R_raw[:, 1] += 0.005  # B is better than A
    R = pd.DataFrame(R_raw, columns=["A", "B"])

    # View: A > B (Contradicts prior)
    views = [{"type": "relative", "asset_high": "A", "asset_low": "B"}]

    p_post = meucci_views(R, views)
    moments = meucci_moments(R.values, p_post)

    mu_a = moments["mu"][0]
    mu_b = moments["mu"][1]

    # After entropy pooling, mu_a should be >= mu_b (within tolerance)
    assert mu_a >= mu_b - 1e-9


def test_meucci_absolute_view():
    np.random.seed(42)
    T, N = 100, 1
    R_raw = np.random.randn(T, N) * 0.01
    R = pd.DataFrame(R_raw, columns=["A"])

    # Current mean is near 0
    target_mu = 0.01  # Within reasonable bounds of random N(0, 0.01)
    views = [{"type": "absolute", "asset": "A", "value": target_mu}]

    p_post = meucci_views(R, views)
    moments = meucci_moments(R.values, p_post)

    mu_post = moments["mu"][0]
    # Check if it's close to the target
    assert np.abs(mu_post - target_mu) < 1e-6


def test_meucci_ranking():
    np.random.seed(42)
    T, N = 100, 3
    R_raw = np.random.randn(T, N) * 0.01
    R = pd.DataFrame(R_raw, columns=["A", "B", "C"])

    # Order: C < B < A
    order = ["C", "B", "A"]
    moments = meucci_ranking(R, order)

    mu = moments["mu"]
    # Indices: A=0, B=1, C=2
    # Expectation: mu[2] < mu[1] < mu[0]
    assert mu[2] < mu[1] < mu[0]


def test_meucci_cv():
    import json

    with open("data/meucci_cv.json", "r") as f:
        cv_data = json.load(f)

    R_raw = np.array(cv_data["input_R"])
    prior_probs = np.array(cv_data["prior_probs"])
    T = len(prior_probs)

    # Reconstruct the equality test case from data/meucci_cv.json
    Aeq = np.vstack([np.ones(T), R_raw[:, 0]])
    beq = np.array([1.0, 0.005])

    res = entropy_prog(prior_probs, Aeq=Aeq, beq=beq)
    expected_p = np.array(cv_data["entropy_prog_eq"]["p_posterior"])

    np.testing.assert_allclose(res["p_"], expected_p, rtol=1e-7, atol=1e-7)


def test_meucci_entropy_pooling_convergence_warning():
    """Verify that entropy pooling returns prior when optimization fails."""
    np.random.seed(42)
    T = 50
    R = np.random.randn(T, 2)
    prior_probs = np.full(T, 1.0 / T)

    # Infeasible views: sum(p)=1 and sum(p)=2
    Aeq = np.vstack([np.ones(T), np.ones(T)])
    beq = np.array([1.0, 2.0])

    # Should not crash, but return prior or best effort
    res = entropy_prog(prior_probs, Aeq=Aeq, beq=beq)
    assert not res["optimizationPerformance"]["converged"]
    # It might return the original probs if it failed completely
    assert len(res["p_"]) == T
