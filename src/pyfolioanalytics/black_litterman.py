"""Black-Litterman posterior moment estimation.

Supports two formulations selected via the ``formulation`` keyword:

``"meucci"`` *(default, matches R's* ``BlackLittermanFormula`` *)*
    Uses the investor's own prior mean **Mu** and covariance **Sigma**
    directly.  No ``tau`` scalar needed — confidence is encoded entirely
    in **Omega**.  Formula (Meucci 2005, eq. 4.41):

    .. code-block::

        BLMu    = Mu + Σ P' (P Σ P' + Ω)^{-1} (q − P Mu)
        BLSigma = Σ  − Σ P' (P Σ P' + Ω)^{-1} P Σ

``"he_litterman"``
    Classic He-Litterman (2002) formulation.  Derives the prior mean as
    implied equilibrium returns ``Pi = lambda * Sigma @ w_mkt`` and uses
    ``tau`` to scale prior uncertainty.

Both formulations share the same **Omega** and **q** defaults, which
mirror R's ``black.litterman()`` function.

For both formulations, **Omega** can also be set to ``"idzorek"`` to
automatically compute the uncertainty matrix from percentage view
confidences using :func:`idzorek_omega`.
"""

from typing import Any, Literal

import numpy as np


def idzorek_omega(
    view_confidences: np.ndarray,
    P: np.ndarray,
    Sigma: np.ndarray,
    tau: float = 1.0,
) -> np.ndarray:
    r"""Compute the view-uncertainty matrix Ω using Idzorek's method.

    Converts percentage view confidences into a diagonal uncertainty
    matrix Ω via the closed-form solution of Walters (2014),
    eq.\ 41 & eq.\ 44:

    .. math::

        \alpha_k = \frac{1 - c_k}{c_k}
        \qquad
        \omega_k = \tau \cdot \alpha_k \cdot P_k \Sigma P_k^\top

    Boundary behaviour:

    * ``conf=1.0`` → ω=0 (complete confidence: posterior equals the view)
    * ``conf=0.5`` → ω = τ·P_kΣP_k’ (matches He-Litterman proportional prior)
    * ``conf→0``  → ω→∞ (no confidence: view is ignored, posterior = prior)

    Parameters
    ----------
    view_confidences:
        1-D array of K floats in ``(0, 1]``.  ``conf_k = 1.0`` means
        100 % confidence in view *k*; ``conf_k = 0.5`` is the
        uninformative baseline that reproduces the He-Litterman
        proportional prior.
    P:
        View (pick) matrix of shape ``(K, N)``.
    Sigma:
        Prior covariance matrix of shape ``(N, N)``.
    tau:
        Prior uncertainty scaling factor (the same ``tau`` used in
        He-Litterman).  Pass ``tau=1.0`` for the Meucci formulation
        where no additional scaling is applied.

    Returns
    -------
    np.ndarray
        Diagonal ``(K, K)`` uncertainty matrix Ω.

    Raises
    ------
    ValueError
        If any confidence is outside ``(0, 1]`` or if the length of
        *view_confidences* does not match the number of views in *P*.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> R = rng.normal(0, 0.01, (120, 4))
    >>> Sigma = np.cov(R.T)
    >>> P = np.array([[1, -1, 0, 0], [0, 0, 1, -1]])
    >>> Omega = idzorek_omega([0.75, 0.50], P, Sigma, tau=1/120)
    >>> Omega.shape
    (2, 2)
    >>> float(np.linalg.det(Omega)) > 0
    True
    """
    view_confidences = np.asarray(view_confidences, dtype=float).ravel()
    P = np.asarray(P, dtype=float)
    K = len(view_confidences)

    if view_confidences.shape[0] != P.shape[0]:
        raise ValueError(
            f"view_confidences length ({K}) must match the number of "
            f"views in P ({P.shape[0]})."
        )
    if np.any(view_confidences <= 0) or np.any(view_confidences > 1):
        raise ValueError(
            "All view_confidences must be in (0, 1]. "
            "Use a small positive value (e.g. 1e-6) instead of exactly 0 "
            "to express near-zero confidence."
        )

    omegas = np.empty(K)
    for k in range(K):
        conf = view_confidences[k]
        alpha = (1.0 - conf) / conf          # Walters (2014) eq. 44
        p_k = P[k : k + 1, :]               # (1, N) — row slice keeps 2-D
        omegas[k] = tau * alpha * (p_k @ Sigma @ p_k.T).item()  # eq. 41

    return np.diag(omegas)



def black_litterman(
    R: np.ndarray,
    P: np.ndarray,
    q: np.ndarray | None = None,
    Mu: np.ndarray | None = None,
    Sigma: np.ndarray | None = None,
    Omega: "np.ndarray | Literal['idzorek'] | None" = None,
    view_confidences: np.ndarray | None = None,
    *,
    formulation: Literal["meucci", "he_litterman"] = "meucci",
    # He-Litterman specific parameters
    w_mkt: np.ndarray | None = None,
    tau: float | None | Literal["auto"] = "auto",
    risk_aversion: float = 2.5,
) -> dict[str, Any]:
    """Compute Black-Litterman posterior moments.

    Parameters
    ----------
    R:
        Historical returns array of shape ``(T, N)``.  Used to compute
        sample estimates when ``Mu`` or ``Sigma`` is ``None``.
    P:
        View (pick) matrix of shape ``(K, N)``.  Each row encodes one
        investor view as a long-short portfolio of assets.
    q:
        View returns vector of shape ``(K,)``.  If ``None``, defaults to
        ``sqrt(diag(P @ Sigma @ P.T))`` — the same default used by R's
        ``black.litterman()``.
    Mu:
        Prior expected return vector of shape ``(N,)``.
        ``None`` → sample mean of ``R``.
    Sigma:
        Prior covariance matrix of shape ``(N, N)``.
        ``None`` → sample covariance of ``R``.
    Omega:
        View uncertainty matrix of shape ``(K, K)``.

        * ``None`` → ``P @ Sigma @ P.T`` (R's proportional-confidence default).
        * ``"idzorek"`` → computed automatically from *view_confidences*
          using :func:`idzorek_omega`; requires *view_confidences* to be
          provided.
        * Any ``(K, K)`` array → used directly.

    view_confidences:
        1-D array of K floats in ``(0, 1]`` representing percentage
        confidence in each view.  Required when ``Omega="idzorek"``;
        ignored otherwise.
    formulation:
        Selects the BL variant:

        * ``"meucci"`` *(default)* — matches R's ``BlackLittermanFormula``
          exactly.  ``tau``, ``w_mkt`` and ``risk_aversion`` are ignored.
        * ``"he_litterman"`` — classic He-Litterman (2002) formula.  Uses
          ``tau``, ``w_mkt``, and ``risk_aversion``.

    w_mkt:
        Market-cap portfolio weights of shape ``(N,)``.
        Only used when ``formulation="he_litterman"``.
        ``None`` → equal weights ``1/N``.
    tau:
        Confidence scalar for ``"he_litterman"``.

        * ``"auto"`` *(default)* — sets ``tau = 1 / T``, the recommendation
          of He & Litterman (2002) and Meucci (2005).
        * A positive float — used directly (e.g. ``tau=0.05`` for the
          legacy default).
        * ``None`` — treated the same as ``"auto"``.

        Ignored when ``formulation="meucci"``.
    risk_aversion:
        Implied risk-aversion coefficient λ.  Used only for
        ``"he_litterman"`` to compute equilibrium returns
        ``Pi = λ · Σ · w_mkt``.

    Returns
    -------
    dict
        * ``"mu"``    — posterior mean, shape ``(N,)``
        * ``"sigma"`` — posterior covariance, shape ``(N, N)``
        * ``"Pi"``    — implied equilibrium returns ``(N,)``
          *(He-Litterman only; absent in Meucci mode)*

    Examples
    --------
    Meucci mode (matches R default):

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> R = rng.normal(0, 0.01, (120, 3))
    >>> P = np.array([[1, -1, 0]])   # asset 0 outperforms asset 1
    >>> q = np.array([0.005])
    >>> res = black_litterman(R, P, q)
    >>> res["mu"].shape
    (3,)
    >>> res["sigma"].shape
    (3, 3)

    Idzorek mode — specify 80 % confidence in a view:

    >>> res_idz = black_litterman(
    ...     R, P, q,
    ...     Omega="idzorek",
    ...     view_confidences=np.array([0.80]),
    ... )
    >>> res_idz["mu"].shape
    (3,)
    """
    R = np.asarray(R, dtype=float)
    T, N = R.shape
    P = np.asarray(P, dtype=float)

    # ── Prior estimation (mirrors R's black.litterman() NULL defaults) ────────
    if Mu is None:
        Mu = np.mean(R, axis=0)
    Mu = np.asarray(Mu, dtype=float).ravel()

    if Sigma is None:
        Sigma = np.cov(R.T, ddof=1)
    Sigma = np.asarray(Sigma, dtype=float)

    # ── Resolve tau early (needed for Idzorek in He-Litterman mode) ───────────
    if tau == "auto" or tau is None:
        tau_val = 1.0 / T
    else:
        tau_val = float(tau)

    # ── Omega: three modes ─────────────────────────────────────────────────────
    PSP = P @ Sigma @ P.T       # K × K — used by both formulations
    if Omega is None:
        # Default: He-Litterman proportional prior (matches R)
        Omega = PSP
    elif isinstance(Omega, str) and Omega == "idzorek":
        # Idzorek shorthand: derive Ω from percentage view confidences.
        # Use tau=1 for Meucci (Sigma is already the full prior covariance;
        # no extra τ scaling); use tau_val for He-Litterman (matches τΣ scale).
        if view_confidences is None:
            raise ValueError(
                "Omega='idzorek' requires view_confidences — a 1-D array of "
                "K floats in (0, 1] specifying percentage confidence in each view."
            )
        tau_idz = tau_val if formulation == "he_litterman" else 1.0
        Omega = idzorek_omega(view_confidences, P, Sigma, tau=tau_idz)
    Omega = np.asarray(Omega, dtype=float)

    if q is None:
        # R: if(is.null(Views)) Views = as.numeric(sqrt(diag(Omega)))
        q = np.sqrt(np.maximum(np.diag(Omega), 0.0))
    q = np.asarray(q, dtype=float).ravel()

    # ── Meucci formulation — exact match to R's BlackLittermanFormula ─────────
    if formulation == "meucci":
        # BLMu    = Mu + Σ P' (P Σ P' + Ω)^{-1} (q − P Mu)
        # BLSigma = Σ  − Σ P' (P Σ P' + Ω)^{-1} P Σ
        A = PSP + Omega          # K × K  (reuse pre-computed PSP)
        innovation = q - P @ Mu  # K
        adj = np.linalg.solve(A, innovation)  # K
        BLMu = Mu + Sigma @ P.T @ adj

        # Use solve for the right-hand side product for numerical stability
        BLSigma = Sigma - Sigma @ P.T @ np.linalg.solve(A.T, P @ Sigma)
        return {"mu": BLMu, "sigma": BLSigma}

    # ── He-Litterman formulation ───────────────────────────────────────────────
    # tau already resolved above (before Omega/Idzorek block)
    if w_mkt is None:
        w_mkt = np.full(N, 1.0 / N)
    w_mkt = np.asarray(w_mkt, dtype=float).ravel()

    Pi = risk_aversion * Sigma @ w_mkt  # implied equilibrium returns (N,)
    tSP = tau_val * Sigma @ P.T         # N × K  (tau-scaled prior × P')
    tPSP = P @ tSP                      # K × K  (P tau Σ P')
    A = tPSP + Omega                    # K × K
    innovation = q - P @ Pi             # K

    BLMu = Pi + tSP @ np.linalg.solve(A, innovation)
    # σ_bl = (1+τ)Σ − τ² Σ P' (P τΣ P' + Ω)^{-1} P Σ
    BLSigma = (1 + tau_val) * Sigma - tau_val**2 * Sigma @ P.T @ np.linalg.solve(
        A.T, P @ Sigma
    )
    return {"mu": BLMu, "sigma": BLSigma, "Pi": Pi}



def black_litterman_tilt(
    w_prior: np.ndarray,
    Sigma: np.ndarray,
    mu_bl: np.ndarray,
    mu_prior: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Tilt portfolio weights towards Black-Litterman views.

    Computes a view-adjusted weight vector by adding the mean-deviation
    term scaled by the inverse covariance (information-ratio tilt):

    .. code-block::

        w_tilted = w_prior + scale · Σ^{-1} (μ_BL − μ_prior)

    This is a *soft* constraint: rather than hard-coding the BL posterior
    into the optimisation objective, the prior weights are nudged in the
    direction suggested by the views.  The caller is responsible for
    re-normalising the result if sum-to-one is required.

    Parameters
    ----------
    w_prior:
        Benchmark (prior) portfolio weights of shape ``(N,)``.
    Sigma:
        Prior covariance matrix of shape ``(N, N)``.
    mu_bl:
        Black-Litterman posterior mean of shape ``(N,)``.
    mu_prior:
        Prior expected return vector of shape ``(N,)``.
    scale:
        Scaling factor for the tilt magnitude.  ``1.0`` applies the
        information-ratio tilt at full strength.

    Returns
    -------
    np.ndarray
        Tilted weights of shape ``(N,)``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> R = rng.normal(0, 0.01, (120, 3))
    >>> Sigma = np.cov(R.T)
    >>> mu_prior = np.mean(R, axis=0)
    >>> P = np.array([[1, -1, 0]])
    >>> q = np.array([0.005])
    >>> res = black_litterman(R, P, q)
    >>> w = black_litterman_tilt(
    ...     w_prior=np.full(3, 1/3),
    ...     Sigma=Sigma, mu_bl=res["mu"], mu_prior=mu_prior,
    ... )
    >>> w.shape
    (3,)
    """
    w_prior  = np.asarray(w_prior, dtype=float).ravel()
    mu_bl    = np.asarray(mu_bl,   dtype=float).ravel()
    mu_prior = np.asarray(mu_prior, dtype=float).ravel()
    delta_mu = mu_bl - mu_prior
    tilt = np.linalg.solve(Sigma, delta_mu)
    return w_prior + scale * tilt
