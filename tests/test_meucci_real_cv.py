import json
import numpy as np
from pyfolioanalytics.meucci import entropy_prog, meucci_moments


def load_real_cv_data():
    with open("data/meucci_real_cv.json", "r") as f:
        return json.load(f)


def test_meucci_mixed_views_real_data():
    data = load_real_cv_data()
    R = np.array(data["input_R"])
    T, N = R.shape
    prior_probs = np.array(data["prior_probs"])

    # Reconstruct constraints as in R script
    # idx_ca=0 (Convertible Arbitrage)
    # idx_cta=1 (CTA Global)
    # idx_ds=2 (Distressed Securities)

    # Aeq: sum(p)=1 and E[R_ca]=0.004
    Aeq = np.zeros((2, T))
    Aeq[0, :] = 1.0
    Aeq[1, :] = R[:, 0]
    beq = np.array([1.0, 0.004])

    # Aineq: E[R_ds - R_cta] <= 0 (i.e. E[R_cta] >= E[R_ds])
    Aineq = (R[:, 2] - R[:, 1]).reshape(1, -1)
    bineq = np.array([0.0])

    res = entropy_prog(prior_probs, A=Aineq, b=bineq, Aeq=Aeq, beq=beq)

    # Verify posterior probabilities
    expected_p = np.array(data["mixed_views"]["p_posterior"])
    np.testing.assert_allclose(res["p_"], expected_p, rtol=1e-6, atol=1e-6)

    # Verify moments
    moments = meucci_moments(R, res["p_"])
    expected_mu = np.array(data["mixed_views"]["mu"])
    expected_sigma = np.array(data["mixed_views"]["sigma"])

    np.testing.assert_allclose(moments["mu"], expected_mu, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(moments["sigma"], expected_sigma, rtol=1e-7, atol=1e-7)
