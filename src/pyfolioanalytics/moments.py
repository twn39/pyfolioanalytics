from dataclasses import dataclass, fields as _dc_fields
from typing import Any, Protocol, runtime_checkable
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .factors import ac_ranking, factor_model_covariance, statistical_factor_model

def clean_returns_boudt(R: pd.DataFrame | np.ndarray, alpha: float = 0.05) -> pd.DataFrame | np.ndarray:
    """
    Robust return cleaning (Winsorization) using the Boudt et al. (2008) method.
    Identifies multivariate outliers using Mahalanobis distance based on MCD 
    (Minimum Covariance Determinant) robust estimates and scales them back to 
    the boundaries of the chi-squared distribution.
    """
    isinstance(R, pd.DataFrame)
    R_vals = R.values if isinstance(R, pd.DataFrame) else np.asarray(R)
    T, N = R_vals.shape
    
    # 1. Robust Mean and Covariance estimation (MCD)
    try:
        from sklearn.covariance import MinCovDet
        # Ensure sufficient observations for MCD, otherwise fallback to standard
        if T > 2 * N:
            mcd = MinCovDet(random_state=42).fit(R_vals)
            mu_mcd = mcd.location_
            cov_mcd = mcd.covariance_
        else:
            raise ValueError("Not enough observations for MCD")
    except Exception as e:
        warnings.warn(f"MCD fitting failed ({str(e)}), falling back to sample moments for Boudt cleaning.")
        mu_mcd = np.mean(R_vals, axis=0)
        cov_mcd = np.cov(R_vals, rowvar=False)
        
    # Calculate pseudo-inverse to handle ill-conditioned covariance
    cov_inv = np.linalg.pinv(cov_mcd)
    
    # 2. Squared Mahalanobis Distance D^2
    diff = R_vals - mu_mcd
    # Vectorized computation of (R_t - mu)^T * Sigma^-1 * (R_t - mu)
    D_sq = np.sum(np.dot(diff, cov_inv) * diff, axis=1)
    
    # 3. Chi-Square threshold (df = N assets)
    threshold = chi2.ppf(1 - alpha, df=N)
    
    # 4. Outlier detection and scaling factor computation
    scaling_factors = np.ones(T)
    outliers = D_sq > threshold
    if np.any(outliers):
        scaling_factors[outliers] = np.sqrt(threshold / D_sq[outliers])
    
    # 5. Winsorization (scaling back outliers)
    R_clean = mu_mcd + diff * scaling_factors[:, np.newaxis]
    
    if isinstance(R, pd.DataFrame):
        return pd.DataFrame(R_clean, index=R.index, columns=R.columns)
    return R_clean

def M3_MM(R: np.ndarray) -> np.ndarray:
    T, N = R.shape
    # Vectorized: M3[i,j,k] = mean(R[:,i] * R[:,j] * R[:,k])
    M3 = np.einsum("ti,tj,tk->ijk", R, R, R) / T
    return M3.reshape(N, N * N)


def M4_MM(R: np.ndarray) -> np.ndarray:
    T, N = R.shape
    # Vectorized: M4[i,j,k,l] = mean(R[:,i] * R[:,j] * R[:,k] * R[:,l])
    M4 = np.einsum("ti,tj,tk,tl->ijkl", R, R, R, R) / T
    return M4.reshape(N, N**3)


def M3_SFM(R: pd.DataFrame, k: int = 1) -> np.ndarray:
    """
    Coskewness matrix estimate via Statistical Factor Model.
    """
    from .factors import statistical_factor_model

    fm = statistical_factor_model(R, k=k)
    B = fm["loadings"].values
    f = fm["factors"].values
    res = fm["residuals"].values
    T, N = R.shape

    # Factor M3
    f_centered = f - np.mean(f, axis=0)
    M3_f = M3_MM(f_centered)

    # Residual M3 (diagonal-like)
    stockM3 = np.sum(res**3, axis=0) / (T - k - 1)

    # S = B * M3_f * (B.T kron B.T)
    Bt = B.T
    S = (B @ M3_f) @ np.kron(Bt, Bt)

    # D residual matrix (N x N^2)
    D = np.zeros((N, N**2))
    for i in range(N):
        D[i, i * N + i] = stockM3[i]

    return S + D


def M4_SFM(R: pd.DataFrame, k: int = 1) -> np.ndarray:
    """
    Cokurtosis matrix estimate via Statistical Factor Model.
    """
    from .factors import statistical_factor_model

    fm = statistical_factor_model(R, k=k)
    B = fm["loadings"].values
    f = fm["factors"].values
    res = fm["residuals"].values
    T, N = R.shape

    # Factor M4
    f_centered = f - np.mean(f, axis=0)
    M4_f = M4_MM(f_centered)

    # Factor M2 (Covariance)
    # R's cov(f) uses T-1
    f2 = np.cov(f, rowvar=False).reshape(k, k)

    # Residual moments
    stockM2 = np.sum(res**2, axis=0) / (T - k - 1)
    stockM4 = np.sum(res**4, axis=0) / (T - k - 1)

    # S = B * M4_f * (B.T kron B.T kron B.T)
    Bt = B.T
    S = (B @ M4_f) @ np.kron(Bt, np.kron(Bt, Bt))

    # D residual matrix (N x N^3)
    # This is complex in MF. For SFM k=1 it's easier.
    # In PA, it calls a C routine. We'll implement the structured residual part.
    # D residual matrix (N x N^3)
    D = np.zeros((N, N**3))

    # Full Kronecker residual terms for SFM (k=1)
    if k == 1:
        # Match residualcokurtosisSF from PortfolioAnalytics
        b = B.flatten()
        f2_val = f2.item()
        s2 = stockM2
        s4 = stockM4

        # We need to fill D[l, i*N*N + j*N + k] = kijkl
        # Following the C code's logic:
        for i in range(N):
            for j in range(N):
                for k_idx in range(N):
                    for l_idx in range(N):
                        kijkl = 0.0
                        if (
                            (i == j)
                            or (i == k_idx)
                            or (i == l_idx)
                            or (j == k_idx)
                            or (j == l_idx)
                            or (k_idx == l_idx)
                        ):
                            if (i == j) and (i == k_idx) and (i == l_idx):
                                kijkl = 6 * b[i] * b[i] * f2_val * s2[i] + s4[i]
                            elif (
                                ((i == j) and (i == k_idx))
                                or ((i == j) and (i == l_idx))
                                or ((i == k_idx) and (i == l_idx))
                                or ((j == k_idx) and (j == l_idx))
                            ):
                                if (i == j) and (i == k_idx):
                                    kijkl = 3 * b[i] * b[l_idx] * f2_val * s2[i]
                                elif (i == j) and (i == l_idx):
                                    kijkl = 3 * b[i] * b[k_idx] * f2_val * s2[i]
                                elif (i == k_idx) and (i == l_idx):
                                    kijkl = 3 * b[i] * b[j] * f2_val * s2[i]
                                elif (j == k_idx) and (j == l_idx):
                                    kijkl = 3 * b[j] * b[i] * f2_val * s2[j]
                            elif (
                                ((i == j) and (k_idx == l_idx))
                                or ((i == k_idx) and (j == l_idx))
                                or ((i == l_idx) and (j == k_idx))
                            ):
                                if (i == j) and (k_idx == l_idx):
                                    kijkl = (
                                        b[i] * b[i] * f2_val * s2[k_idx]
                                        + b[k_idx] * b[k_idx] * f2_val * s2[i]
                                        + s2[i] * s2[k_idx]
                                    )
                                elif (i == k_idx) and (j == l_idx):
                                    kijkl = (
                                        b[i] * b[i] * f2_val * s2[j]
                                        + b[j] * b[j] * f2_val * s2[i]
                                        + s2[i] * s2[j]
                                    )
                                elif (i == l_idx) and (j == k_idx):
                                    kijkl = (
                                        b[i] * b[i] * f2_val * s2[j]
                                        + b[j] * b[j] * f2_val * s2[i]
                                        + s2[i] * s2[j]
                                    )
                            else:
                                if i == j:
                                    kijkl = b[k_idx] * b[l_idx] * f2_val * s2[i]
                                elif i == k_idx:
                                    kijkl = b[j] * b[l_idx] * f2_val * s2[i]
                                elif i == l_idx:
                                    kijkl = b[j] * b[k_idx] * f2_val * s2[i]
                                elif j == k_idx:
                                    kijkl = b[i] * b[l_idx] * f2_val * s2[j]
                                elif j == l_idx:
                                    kijkl = b[i] * b[k_idx] * f2_val * s2[j]
                                elif k_idx == l_idx:
                                    kijkl = b[i] * b[j] * f2_val * s2[k_idx]

                        D[l_idx, i * N * N + j * N + k_idx] = kijkl
    else:
        # Multi-factor residual approximation
        for i in range(N):
            D[i, i * N**2 + i * N + i] = stockM4[i]

    return S + D


def shrink_comoments(
    M_sample: np.ndarray, M_target: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    return (1 - alpha) * M_sample + alpha * M_target


def ewma_moments(R: np.ndarray, span: int = 36) -> dict[str, Any]:
    """
    Calculate Exponentially Weighted Moving Average (EWMA) mean and covariance.
    """
    alpha = 2.0 / (span + 1)
    T, N = R.shape
    weights = (1 - alpha) ** np.arange(T - 1, -1, -1)
    weights /= weights.sum()

    mu = np.average(R, weights=weights, axis=0)
    R_centered = R - mu
    # unbiased-like normalization can be done, but keeping it simple with weights
    cov = (weights * R_centered.T) @ R_centered
    return {"mu": mu.reshape(-1, 1), "sigma": cov}


def semi_covariance(R: np.ndarray, benchmark: float = 0.0) -> np.ndarray:
    """
    Calculate semi-covariance matrix (downside covariance), penalizing returns below benchmark.
    """
    R_down = np.minimum(R - benchmark, 0.0)
    T = R.shape[0]
    return (R_down.T @ R_down) / T


def ema_returns(R: pd.DataFrame, span: int = 252) -> np.ndarray:
    """
    Calculate the exponentially-weighted mean of historical returns.
    """
    ewm_mean = R.ewm(span=span).mean().iloc[-1]
    return ewm_mean.values.reshape(-1, 1)


def capm_returns(
    R: pd.DataFrame,
    market_returns: pd.Series | None = None,
    market_caps: pd.Series | dict[str, float] | None = None,
    risk_free_rate: float = 0.0,
) -> np.ndarray:
    """
    Calculate the expected returns based on the Capital Asset Pricing Model (CAPM).
    Matches PyPfOpt's capm_return logic strictly.
    """
    returns = R.copy()
    
    if market_returns is not None:
        if isinstance(market_returns, pd.DataFrame):
            market_returns = market_returns.iloc[:, 0]
    else:
        # Construct proxy for market
        if market_caps is not None:
            mc = pd.Series(market_caps)
            mc = mc.reindex(R.columns).fillna(0.0)
            if mc.sum() > 0:
                weights = mc / mc.sum()
            else:
                weights = pd.Series(1.0 / len(R.columns), index=R.columns)
            market_returns = R.dot(weights)
        else:
            market_returns = R.mean(axis=1)

    returns["mkt"] = market_returns
    cov = returns.cov()
    
    # Beta = Cov(R_i, R_m) / Var(R_m)
    if cov.loc["mkt", "mkt"] > 0:
        betas = cov["mkt"] / cov.loc["mkt", "mkt"]
    else:
        betas = pd.Series(0.0, index=returns.columns)
        
    betas = betas.drop("mkt")

    # Assuming daily returns, compounding to match PyPfOpt's annualized mkt_mean_ret logic
    # BUT we want to return raw scale to match PyFolioAnalytics API. We use raw means.
    # PyPfOpt allows toggling compounding. We'll stick to raw mean.
    mkt_mean_ret = (1 + returns["mkt"]).prod() ** (1.0 / returns["mkt"].count()) - 1.0
    
    # Expected return = Rf + Beta * (E[Rm] - Rf)
    expected_returns = risk_free_rate + betas * (mkt_mean_ret - risk_free_rate)
    
    return expected_returns.values.reshape(-1, 1)


def shrunk_covariance(
    R: pd.DataFrame, 
    method: str = "ledoit_wolf", 
    shrinkage_target: str = "constant_variance"
) -> np.ndarray:
    """
    Native implementation of advanced covariance shrinkage (Ledoit-Wolf & OAS).
    """
    from sklearn.covariance import LedoitWolf, OAS
    
    X = np.nan_to_num(R.values)
    t, n = X.shape

    if method == "oas":
        oas = OAS(assume_centered=False).fit(X)
        return oas.covariance_
        
    elif method == "ledoit_wolf":
        if shrinkage_target == "constant_variance":
            # Scikit-learn's default implementation
            lw = LedoitWolf(assume_centered=False).fit(X)
            return lw.covariance_
            
        elif shrinkage_target == "constant_correlation":
            # Native implementation matching Ledoit & Wolf (2003) / PyPfOpt
            S = np.cov(X, rowvar=False)
            
            var = np.diag(S).reshape(-1, 1)
            std = np.sqrt(var)
            _var = np.tile(var, (n,))
            _std = np.tile(std, (n,))
            
            with np.errstate(divide='ignore', invalid='ignore'):
                cor_mat = S / (_std * _std.T)
                cor_mat[np.isnan(cor_mat) | np.isinf(cor_mat)] = 0.0
                
            r_bar = (np.sum(cor_mat) - n) / (n * (n - 1)) if n > 1 else 1.0
            
            F = r_bar * (_std * _std.T)
            F[np.eye(n) == 1] = var.reshape(-1)
            
            Xm = X - X.mean(axis=0)
            y = Xm**2
            
            # Estimate pi
            pi_mat = np.dot(y.T, y) / t - 2 * np.dot(Xm.T, Xm) * S / t + S**2
            pi_hat = np.sum(pi_mat)
            
            # Theta matrix, expanded term by term
            term1 = np.dot((Xm**3).T, Xm) / t
            help_ = np.dot(Xm.T, Xm) / t
            help_diag = np.diag(help_)
            term2 = np.tile(help_diag, (n, 1)).T * S
            term3 = help_ * _var
            term4 = _var * S
            
            theta_mat = term1 - term2 - term3 + term4
            theta_mat[np.eye(n) == 1] = np.zeros(n)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                inv_std = np.where(std > 1e-10, 1.0 / std, 0.0)
                
            rho_hat = np.sum(np.diag(pi_mat)) + r_bar * np.sum(
                np.dot(inv_std, std.T) * theta_mat
            )
            
            # Estimate gamma
            gamma_hat = np.linalg.norm(S - F, "fro") ** 2
            
            # Compute shrinkage constant
            if gamma_hat < 1e-10:
                delta = 0.0
            else:
                kappa_hat = (pi_hat - rho_hat) / gamma_hat
                delta = max(0.0, min(1.0, kappa_hat / t))
                
            return delta * F + (1.0 - delta) * S
            
        else:
            raise ValueError(f"Unknown shrinkage target: {shrinkage_target}")
    else:
        raise ValueError(f"Unknown shrinkage method: {method}")


def ccc_garch_moments(R: np.ndarray, mu: np.ndarray | None = None) -> dict[str, Any]:
    """
    Constant Conditional Correlation (CCC) GARCH Moment Model.
    Equivalent to PortfolioAnalytics::CCCgarch.MM.
    """
    import warnings

    from arch import arch_model

    T, N = R.shape
    if mu is None:
        mu = np.mean(R, axis=0)

    R_centered = R - mu
    S = np.zeros((T, N))
    nextS = np.zeros(N)

    for i in range(N):
        # Scale returns by 100 for stability (arch library recommendation)
        scale_factor = 100.0
        y = R_centered[:, i] * scale_factor

        # Fit GARCH(1,1)
        model = arch_model(y, vol="GARCH", p=1, q=1, mean="Zero", dist="normal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(disp="off", show_warning=False)

        # alpha1 check (on scaled parameters)
        alpha1 = res.params.get("alpha[1]", 0.0)

        if alpha1 < 0.01:
            sigmat_scaled = np.full(T, np.std(y))
            nextSt_scaled = np.std(y)
        else:
            sigmat_scaled = res.conditional_volatility
            forecast = res.forecast(horizon=1)
            nextSt_scaled = np.sqrt(forecast.variance.values[-1, 0])

        # De-scale results
        S[:, i] = sigmat_scaled / scale_factor
        nextS[i] = nextSt_scaled / scale_factor

    # Standardized residuals
    U = R_centered / S

    # Constant Correlation Matrix
    Rcor = np.corrcoef(U, rowvar=False)

    # Conditional Covariance Matrix for next period
    D = np.diag(nextS)
    sigma = D @ Rcor @ D

    # Rescale U for higher order moments matching R
    # uncS = sqrt(diag(cov(U)))
    uncS = np.std(U, axis=0)
    U_rescaled = U * (nextS / uncS)

    return {
        "mu": mu.reshape(-1, 1),
        "sigma": sigma,
        "m3": M3_MM(U_rescaled),
        "m4": M4_MM(U_rescaled),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MomentConfig — typed replacement for **kwargs scatter-gun
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MomentConfig:
    """Typed configuration for :func:`set_portfolio_moments`.

    Replaces the untyped ``**kwargs`` API.  All parameters are documented
    and type-checked.  Unknown kwargs passed via the legacy path emit a
    :class:`DeprecationWarning`.
    """
    # Dispatch
    method: str = "sample"
    sigma_method: str | None = None      # falls back to method if None
    mu_method: str | None = None         # falls back to method if None
    comoment_method: str = "sample"

    # Return cleaning
    clean_returns: str | None = None
    clean_alpha: float = 0.05

    # Shrinkage
    shrinkage_target: str = "constant_variance"

    # Factor model / comoments
    k: int = 1
    comoment_alpha: float = 0.0

    # EWMA
    span: int = 36
    ema_span: int = 252

    # AC Ranking
    order: list[str] | None = None

    # Semi-covariance
    benchmark: float = 0.0

    # CAPM
    market_returns: "pd.Series | None" = None
    market_caps: Any = None
    risk_free_rate: float = 0.0

    # Black-Litterman
    P: "np.ndarray | None" = None
    q: "np.ndarray | None" = None
    Omega: "np.ndarray | str | None" = None   # accepts 'idzorek'
    view_confidences: "np.ndarray | None" = None  # required when Omega='idzorek'
    Mu: "np.ndarray | None" = None
    Sigma: "np.ndarray | None" = None
    bl_formulation: str = "meucci"
    tau: Any = "auto"
    risk_aversion: float = 2.5
    w_mkt: "np.ndarray | None" = None

    # Denoising (RMT)
    denoise_method: str = "fixed"

    # Meucci / Entropy Pooling
    prior_probs: "np.ndarray | None" = None
    Aeq: "np.ndarray | None" = None
    beq: "np.ndarray | None" = None

    @classmethod
    def from_kwargs(cls, method: str = "sample", **kwargs: Any) -> "MomentConfig":
        """Build a :class:`MomentConfig` from legacy ``**kwargs``.

        Only pass kwargs that are actually moment-estimation parameters;
        the caller is responsible for filtering out solver/optimizer kwargs
        *before* calling this method.  Any field not recognised as a
        :class:`MomentConfig` attribute emits a :class:`DeprecationWarning`.
        """
        known = {f.name for f in _dc_fields(cls)}
        filtered: dict[str, Any] = {}
        unknown: list[str] = []
        for k, v in kwargs.items():
            if k in known:
                filtered[k] = v
            else:
                unknown.append(k)
        if unknown:
            warnings.warn(
                f"set_portfolio_moments received unknown keyword argument(s) "
                f"{unknown!r}; they are ignored. "
                f"Pass a MomentConfig object instead of **kwargs to silence this warning.",
                DeprecationWarning,
                stacklevel=3,
            )
        return cls(method=method, **filtered)



# ─────────────────────────────────────────────────────────────────────────────
# Protocols
# ─────────────────────────────────────────────────────────────────────────────

_FitResult = "np.ndarray | dict[str, Any]"


@runtime_checkable
class CovarianceEstimator(Protocol):
    """Structural interface for covariance matrix estimators.

    ``fit()`` may return either:

    * ``np.ndarray`` — the (N, N) covariance matrix only.
    * ``dict`` — a multi-moment result containing at minimum ``"sigma"``;
      may also contain ``"mu"`` (side-effect mean) and control flags
      ``"_mu_priority"`` (bool, BL: mu cannot be overridden) and
      ``"_mu_from_cov"`` (bool, GARCH/EWMA/Meucci: mu as side-effect).
    """

    def fit(self, R: "pd.DataFrame") -> _FitResult:  # type: ignore[type-arg]
        ...


@runtime_checkable
class ReturnEstimator(Protocol):
    """Structural interface for expected-return estimators."""

    def fit(self, R: "pd.DataFrame") -> "np.ndarray":
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Registries
# ─────────────────────────────────────────────────────────────────────────────

_COV_REGISTRY: dict[str, type] = {}
_MU_REGISTRY: dict[str, type] = {}


def register_cov_estimator(*names: str):
    """Class decorator that registers a covariance estimator under *names*."""
    def decorator(cls: type) -> type:
        for name in names:
            _COV_REGISTRY[name] = cls
        return cls
    return decorator


def register_mu_estimator(*names: str):
    """Class decorator that registers a return estimator under *names*."""
    def decorator(cls: type) -> type:
        for name in names:
            _MU_REGISTRY[name] = cls
        return cls
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Covariance estimators
# ─────────────────────────────────────────────────────────────────────────────

@register_cov_estimator("sample")
class SampleCovarianceEstimator:
    def __init__(self, cfg: MomentConfig) -> None: ...
    def fit(self, R: pd.DataFrame) -> np.ndarray:
        return R.cov().values


@register_cov_estimator("shrinkage", "ledoit_wolf", "oas")
class ShrinkageCovarianceEstimator:
    def __init__(self, cfg: MomentConfig) -> None:
        self.target = cfg.shrinkage_target
        self.sigma_method = cfg.sigma_method or cfg.method

    def fit(self, R: pd.DataFrame) -> np.ndarray:
        # Map legacy "shrinkage" → ledoit_wolf / oas
        if self.sigma_method == "shrinkage":
            method_arg = "oas" if self.target == "oas" else "ledoit_wolf"
            target = "constant_variance" if self.target == "identity" else self.target
        else:
            method_arg = self.sigma_method
            target = self.target
        return shrunk_covariance(R, method=method_arg, shrinkage_target=target)


@register_cov_estimator("factor_model")
class FactorModelCovarianceEstimator:
    def __init__(self, cfg: MomentConfig) -> None:
        self.k = cfg.k

    def fit(self, R: pd.DataFrame) -> np.ndarray:
        fm = statistical_factor_model(R, k=self.k)
        return factor_model_covariance(fm)


@register_cov_estimator("robust", "mcd")
class RobustCovarianceEstimator:
    def __init__(self, cfg: MomentConfig) -> None: ...

    def fit(self, R: pd.DataFrame) -> dict[str, Any]:
        from sklearn.covariance import MinCovDet
        mcd = MinCovDet(random_state=42).fit(R.values)
        return {
            "sigma": mcd.covariance_,
            "mu": mcd.location_.reshape(-1, 1),
            "_mu_from_cov": True,
        }


@register_cov_estimator("denoised")
class DenoisedCovarianceEstimator:
    def __init__(self, cfg: MomentConfig) -> None:
        self.denoise_method = cfg.denoise_method

    def fit(self, R: pd.DataFrame) -> np.ndarray:
        from .rmt import denoise_covariance
        T, N = R.shape
        sigma = R.cov().values
        return denoise_covariance(sigma, T / N, method=self.denoise_method)


@register_cov_estimator("garch")
class GARCHCovarianceEstimator:
    def __init__(self, cfg: MomentConfig) -> None: ...

    def fit(self, R: pd.DataFrame) -> dict[str, Any]:
        res = ccc_garch_moments(R.values)
        return {"sigma": res["sigma"], "mu": res["mu"], "_mu_from_cov": True}


@register_cov_estimator("ewma")
class EWMACovarianceEstimator:
    def __init__(self, cfg: MomentConfig) -> None:
        self.span = cfg.span

    def fit(self, R: pd.DataFrame) -> dict[str, Any]:
        res = ewma_moments(R.values, span=self.span)
        return {"sigma": res["sigma"], "mu": res["mu"], "_mu_from_cov": True}


@register_cov_estimator("semi_covariance")
class SemiCovarianceEstimator:
    def __init__(self, cfg: MomentConfig) -> None:
        self.benchmark = cfg.benchmark

    def fit(self, R: pd.DataFrame) -> np.ndarray:
        return semi_covariance(R.values, benchmark=self.benchmark)


@register_cov_estimator("black_litterman")
class BlackLittermanEstimator:
    """BL always produces both sigma and mu; its mu takes priority over any
    separate mu-method request, mirroring R's ``portfolio.moments.bl``."""

    def __init__(self, cfg: MomentConfig) -> None:
        self.cfg = cfg

    def fit(self, R: pd.DataFrame) -> dict[str, Any]:
        from .black_litterman import black_litterman as _bl
        cfg = self.cfg
        asset_names = list(R.columns)
        P = cfg.P if cfg.P is not None else np.ones((1, len(asset_names)))
        res = _bl(
            R.values, P=P, q=cfg.q,
            Mu=cfg.Mu, Sigma=cfg.Sigma, Omega=cfg.Omega,
            view_confidences=cfg.view_confidences,
            formulation=cfg.bl_formulation,
            tau=cfg.tau, risk_aversion=cfg.risk_aversion, w_mkt=cfg.w_mkt,
        )
        return {
            "sigma": res["sigma"],
            "mu": res["mu"].reshape(-1, 1),
            "_mu_from_cov": True,
            "_mu_priority": True,   # BL mu cannot be overridden by a mu-method
        }


@register_cov_estimator("meucci")
class MeucciCovarianceEstimator:
    def __init__(self, cfg: MomentConfig) -> None:
        self.cfg = cfg

    def fit(self, R: pd.DataFrame) -> dict[str, Any]:
        from .meucci import entropy_pooling, meucci_moments
        T = R.shape[0]
        prior = self.cfg.prior_probs if self.cfg.prior_probs is not None else np.full(T, 1.0 / T)
        p = entropy_pooling(prior, Aeq=self.cfg.Aeq, beq=self.cfg.beq)
        res = meucci_moments(R.values, p)
        return {"sigma": res["sigma"], "mu": res["mu"], "_mu_from_cov": True}


@register_cov_estimator("ac_ranking")
class ACRankingCovarianceEstimator:
    """AC Ranking only affects the mean; covariance is sample."""
    def __init__(self, cfg: MomentConfig) -> None: ...

    def fit(self, R: pd.DataFrame) -> np.ndarray:
        return R.cov().values


# ─────────────────────────────────────────────────────────────────────────────
# Return estimators
# ─────────────────────────────────────────────────────────────────────────────

@register_mu_estimator(
    "sample", "historical", "semi_covariance",
    "shrinkage", "denoised", "factor_model",
    # Methods below produce sigma+mu together; if explicitly requested as
    # mu_method they fall back to the sample mean (matching legacy behaviour).
    "garch", "ewma", "meucci", "robust", "mcd", "black_litterman",
)
class SampleReturnEstimator:
    def __init__(self, cfg: MomentConfig) -> None: ...

    def fit(self, R: pd.DataFrame) -> np.ndarray:
        return R.mean().values.reshape(-1, 1)


@register_mu_estimator("ema")
class EMAReturnEstimator:
    def __init__(self, cfg: MomentConfig) -> None:
        self.span = cfg.ema_span

    def fit(self, R: pd.DataFrame) -> np.ndarray:
        return ema_returns(R, span=self.span)


@register_mu_estimator("capm")
class CAPMReturnEstimator:
    def __init__(self, cfg: MomentConfig) -> None:
        self.cfg = cfg

    def fit(self, R: pd.DataFrame) -> np.ndarray:
        return capm_returns(
            R,
            market_returns=self.cfg.market_returns,
            market_caps=self.cfg.market_caps,
            risk_free_rate=self.cfg.risk_free_rate,
        )


@register_mu_estimator("ac_ranking")
class ACRankingReturnEstimator:
    def __init__(self, cfg: MomentConfig) -> None:
        self.order = cfg.order

    def fit(self, R: pd.DataFrame) -> np.ndarray:
        if self.order is None:
            raise ValueError("Method 'ac_ranking' requires an 'order' argument in MomentConfig.")
        return ac_ranking(R, self.order).reshape(-1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# set_portfolio_moments — thin dispatcher (backward-compatible)
# ─────────────────────────────────────────────────────────────────────────────

def set_portfolio_moments(
    R: pd.DataFrame,
    portfolio: Any,
    method: str = "sample",
    config: MomentConfig | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Estimate portfolio moments (μ, Σ, and optionally M3/M4).

    **Legacy style** (fully backward-compatible)::

        set_portfolio_moments(R, port, method="shrinkage",
                              shrinkage_target="constant_correlation")

    **New typed style** (preferred; silences DeprecationWarnings)::

        cfg = MomentConfig(method="shrinkage",
                           shrinkage_target="constant_correlation")
        set_portfolio_moments(R, port, config=cfg)

    Parameters
    ----------
    R:
        Returns ``DataFrame`` (T × N).
    portfolio:
        A :class:`~pyfolioanalytics.portfolio.Portfolio` instance.
    method:
        Default estimation method when ``sigma_method`` / ``mu_method``
        are not separately specified.
    config:
        Typed configuration object.  If ``None``, one is built from
        ``method`` and ``**kwargs`` (legacy path).
    **kwargs:
        Legacy keyword arguments.  Known fields are forwarded to
        :class:`MomentConfig`; unknown fields emit a
        :class:`DeprecationWarning`.
    """
    # Build config from legacy kwargs if not provided
    if config is None:
        config = MomentConfig.from_kwargs(method=method, **kwargs)

    # ── Asset filtering ────────────────────────────────────────────────────
    if hasattr(portfolio, "root"):
        portfolio = portfolio.root
    asset_names = list(portfolio.assets.keys())
    R_filtered = R[asset_names]

    # ── Return cleaning ────────────────────────────────────────────────────
    if config.clean_returns == "boudt":
        R_filtered = pd.DataFrame(
            clean_returns_boudt(R_filtered, alpha=config.clean_alpha),
            columns=R_filtered.columns,
            index=R_filtered.index,
        )

    moments: dict[str, Any] = {}

    # ── Covariance estimation ──────────────────────────────────────────────
    sigma_method = config.sigma_method or config.method
    cov_cls = _COV_REGISTRY.get(sigma_method)
    if cov_cls is None:
        raise NotImplementedError(
            f"Covariance method '{sigma_method}' is not registered. "
            f"Available: {sorted(_COV_REGISTRY)}"
        )
    cov_result = cov_cls(config).fit(R_filtered)

    # Combined estimators return a dict with control flags
    if isinstance(cov_result, dict):
        mu_priority = bool(cov_result.pop("_mu_priority", False))
        mu_from_cov = bool(cov_result.pop("_mu_from_cov", False))
        moments.update(cov_result)
    else:
        moments["sigma"] = cov_result
        mu_priority = False
        mu_from_cov = False

    # ── Return estimation ──────────────────────────────────────────────────
    # Priority rules (matching legacy behaviour exactly):
    #   1. BL mu_priority=True  → always use BL mu, never override.
    #   2. Combined estimator set mu AND no explicit mu_method override
    #      → use the side-effect mu (GARCH/EWMA/Meucci convention).
    #   3. Otherwise → call the registered mu estimator.
    mu_method = config.mu_method or config.method
    do_mu_estimation = not mu_priority and (
        not mu_from_cov or config.mu_method is not None
    )
    if do_mu_estimation:
        mu_cls = _MU_REGISTRY.get(mu_method)
        if mu_cls is not None:
            moments["mu"] = mu_cls(config).fit(R_filtered)
        elif "mu" not in moments:
            # Ultimate fallback: sample mean
            moments["mu"] = R_filtered.mean().values.reshape(-1, 1)

    # ── Sigma fallback ─────────────────────────────────────────────────────
    if "sigma" not in moments:
        moments["sigma"] = R_filtered.cov().values

    # ── Higher-order moments (M3 / M4) ────────────────────────────────────
    # Only computed for modified (Cornish-Fisher) VaR/ES objectives to
    # avoid O(T·N⁴) work when not needed.
    needs_m3_m4 = any(
        obj["name"] in ("VaR", "ES", "mVaR", "mES")
        and obj.get("arguments", {}).get("method", "gaussian") == "modified"
        for obj in portfolio.objectives
        if obj.get("enabled", True)
    )
    if needs_m3_m4:
        R_centered = R_filtered.values - moments["mu"].T
        cm_method = config.comoment_method
        alpha = config.comoment_alpha
        k = config.k
        if cm_method == "sample":
            moments["m3"] = M3_MM(R_centered)
            moments["m4"] = M4_MM(R_centered)
        elif cm_method == "factor_model":
            moments["m3"] = M3_SFM(R_filtered, k=k)
            moments["m4"] = M4_SFM(R_filtered, k=k)
        elif cm_method == "shrinkage":
            moments["m3"] = shrink_comoments(M3_MM(R_centered), M3_SFM(R_filtered, k=k), alpha=alpha)
            moments["m4"] = shrink_comoments(M4_MM(R_centered), M4_SFM(R_filtered, k=k), alpha=alpha)

    return moments

