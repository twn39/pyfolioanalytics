import numpy as np
from typing import List, Tuple, Dict, Any, Optional



class CLA:
    """
    Critical Line Algorithm (CLA) for Mean-Variance Optimization.
    Based on the implementation by Marcos Lopez de Prado.
    """

    def __init__(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ):
        self.mu = expected_returns.reshape(-1, 1)
        self.sigma = cov_matrix
        self.lb = lower_bounds.reshape(-1, 1)
        self.ub = upper_bounds.reshape(-1, 1)
        self.n = len(self.mu)

        self.w = []  # solution weights at turning points
        self.ls = []  # lambdas at turning points
        self.g = []  # gammas at turning points
        self.f = []  # free sets at turning points

    @staticmethod
    def _infnone(x):
        return float("-inf") if x is None else x

    def _init_algo(self) -> Tuple[List[int], np.ndarray]:
        # Form structured array of (id, mu)
        idx = np.argsort(self.mu.flatten())

        # 3) First free weight
        # Start with all at lower bounds
        i, w = self.n, np.copy(self.lb)
        while np.sum(w) < 1.0 and i > 0:
            i -= 1
            idx_i = idx[i]
            w[idx_i] = self.ub[idx_i]

        # Adjust last modified asset to meet sum(w) = 1
        if np.sum(w) > 1.0:
            w[idx[i]] += 1.0 - np.sum(w)

        return [idx[i]], w

    def _get_matrices(
        self, f: List[int], w: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        b = list(set(range(self.n)) - set(f))
        covarF = self.sigma[np.ix_(f, f)]
        meanF = self.mu[f]
        covarFB = self.sigma[np.ix_(f, b)]
        wB = w[b]
        return covarF, covarFB, meanF, wB

    def _compute_bi(self, c, bi):
        if c > 0:
            return bi[1]
        if c < 0:
            return bi[0]
        return bi[0]

    def _compute_lambda(
        self,
        covarF_inv: np.ndarray,
        covarFB: np.ndarray,
        meanF: np.ndarray,
        wB: np.ndarray,
        i: int,
        bi: Any,
    ) -> Tuple[Optional[float], Optional[float]]:
        onesF = np.ones((len(meanF), 1))
        c1 = onesF.T @ covarF_inv @ onesF
        c2 = covarF_inv @ meanF
        c3 = onesF.T @ covarF_inv @ meanF
        c4 = covarF_inv @ onesF

        c = -c1 * c2[i] + c3 * c4[i]
        c_val = c.item()
        if abs(c_val) < 1e-12:
            return None, None

        if isinstance(bi, list):
            bi = self._compute_bi(c_val, bi)

        if len(wB) == 0:
            res = (c4[i] - c1 * bi) / c
        else:
            onesB = np.ones((len(wB), 1))
            l1 = onesB.T @ wB
            l2 = covarF_inv @ covarFB
            l3 = l2 @ wB
            l4 = onesF.T @ l3
            res = ((1 - l1 + l4) * c4[i] - c1 * (bi + l3[i])) / c
        return float(res.item()), float(bi)

    def _compute_w(
        self,
        covarF_inv: np.ndarray,
        covarFB: np.ndarray,
        meanF: np.ndarray,
        wB: np.ndarray,
        lam: float,
    ) -> Tuple[np.ndarray, float]:
        onesF = np.ones((len(meanF), 1))
        g1 = onesF.T @ covarF_inv @ meanF
        g2 = onesF.T @ covarF_inv @ onesF

        if len(wB) == 0:
            g = -lam * g1 / g2 + 1 / g2
            w1 = np.zeros(onesF.shape)
        else:
            onesB = np.ones((len(wB), 1))
            g3 = onesB.T @ wB
            g4 = covarF_inv @ covarFB
            w1 = g4 @ wB
            g5 = onesF.T @ w1
            g = -lam * g1 / g2 + (1 - g3 + g5) / g2

        g_val = float(g.item())
        w2 = covarF_inv @ onesF
        w3 = covarF_inv @ meanF
        wF = -w1 + g_val * w2 + lam * w3
        return wF, g_val

    def solve(self):
        f, w = self._init_algo()
        self.w.append(np.copy(w))
        self.ls.append(None)
        self.g.append(None)
        self.f.append(f[:])

        while True:
            # Case A: Bound one free weight
            l_in = None
            if len(f) > 1:
                covarF, covarFB, meanF, wB = self._get_matrices(f, w)
                covarF_inv = np.linalg.inv(covarF)
                for j, idx in enumerate(f):
                    lam, bi = self._compute_lambda(
                        covarF_inv,
                        covarFB,
                        meanF,
                        wB,
                        j,
                        [self.lb[idx].item(), self.ub[idx].item()],
                    )
                    if self._infnone(lam) > self._infnone(l_in):
                        l_in, i_in, bi_in = lam, idx, bi

            # Case B: Free one bounded weight
            l_out = None
            b = list(set(range(self.n)) - set(f))
            if len(b) > 0:
                for idx in b:
                    f_temp = f + [idx]
                    covarF, covarFB, meanF, wB = self._get_matrices(f_temp, w)
                    covarF_inv = np.linalg.inv(covarF)
                    lam, bi = self._compute_lambda(
                        covarF_inv, covarFB, meanF, wB, len(f_temp) - 1, w[idx].item()
                    )

                    if (
                        self.ls[-1] is None or lam < self.ls[-1]
                    ) and lam > self._infnone(l_out):
                        l_out, i_out = lam, idx

            if self._infnone(l_in) < 0 and self._infnone(l_out) < 0:
                # Minimum Variance Solution
                self.ls.append(0.0)
                covarF, covarFB, meanF, wB = self._get_matrices(f, w)
                covarF_inv = np.linalg.inv(covarF)
                wF, g = self._compute_w(
                    covarF_inv, covarFB, np.zeros(meanF.shape), wB, 0.0
                )
            else:
                if self._infnone(l_in) > self._infnone(l_out):
                    self.ls.append(l_in)
                    f.remove(i_in)
                    w[i_in] = bi_in
                else:
                    self.ls.append(l_out)
                    f.append(i_out)
                covarF, covarFB, meanF, wB = self._get_matrices(f, w)
                covarF_inv = np.linalg.inv(covarF)
                wF, g = self._compute_w(covarF_inv, covarFB, meanF, wB, self.ls[-1])

            for j, idx in enumerate(f):
                w[idx] = wF[j]

            self.w.append(np.copy(w))
            self.g.append(g)
            self.f.append(f[:])

            if self.ls[-1] == 0:
                break

        self._purge_num_err(1e-10)
        self._purge_excess()

    def _purge_num_err(self, tol: float):
        i = 0
        while i < len(self.w):
            w = self.w[i]
            if (
                abs(np.sum(w) - 1.0) > tol
                or np.any(w < self.lb - tol)
                or np.any(w > self.ub + tol)
            ):
                del self.w[i], self.ls[i], self.g[i], self.f[i]
            else:
                i += 1

    def _purge_excess(self):
        i = 0
        while i < len(self.w) - 1:
            mu = (self.w[i].T @ self.mu).item()
            j = i + 1
            removed = False
            while j < len(self.w):
                mu_next = (self.w[j].T @ self.mu).item()
                if mu < mu_next:
                    del self.w[i], self.ls[i], self.g[i], self.f[i]
                    removed = True
                    break
                j += 1
            if not removed:
                i += 1

    def max_sharpe(self, risk_free_rate: float = 0.0) -> np.ndarray:
        if not self.w:
            self.solve()

        def sr_func(alpha, w0, w1):
            w = alpha * w0 + (1 - alpha) * w1
            ret = (w.T @ self.mu).item() - risk_free_rate
            vol = np.sqrt((w.T @ self.sigma @ w).item())
            if vol < 1e-12:
                return 0.0
            return -(ret / vol)  # Minimize negative SR

        from scipy.optimize import minimize_scalar

        best_w = self.w[0]
        max_sr = -np.inf

        for i in range(len(self.w) - 1):
            res = minimize_scalar(
                sr_func,
                bounds=(0, 1),
                args=(self.w[i], self.w[i + 1]),
                method="bounded",
            )
            w_opt = res.x * self.w[i] + (1 - res.x) * self.w[i + 1]
            sr = -res.fun
            if sr > max_sr:
                max_sr = sr
                best_w = w_opt
        return best_w.flatten()

    def min_volatility(self) -> np.ndarray:
        if not self.w:
            self.solve()
        vols = [np.sqrt((w.T @ self.sigma @ w).item()) for w in self.w]
        return self.w[np.argmin(vols)].flatten()

    def efficient_frontier(
        self, points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        if not self.w:
            self.solve()
        mu_list, sigma_list, weights_list = [], [], []

        n_segments = len(self.w) - 1
        if n_segments <= 0:
            w = self.w[0]
            return (
                np.array([(w.T @ self.mu).item()]),
                np.array([np.sqrt((w.T @ self.sigma @ w).item())]),
                [w.flatten()],
            )

        points_per_segment = max(2, points // n_segments)

        for i in range(n_segments):
            alphas = np.linspace(0, 1, points_per_segment)
            if i < n_segments - 1:
                alphas = alphas[:-1]  # avoid duplicate points

            for alpha in alphas:
                w = alpha * self.w[i + 1] + (1 - alpha) * self.w[i]
                weights_list.append(w.flatten())
                mu_list.append((w.T @ self.mu).item())
                sigma_list.append(np.sqrt((w.T @ self.sigma @ w).item()))

        return np.array(mu_list), np.array(sigma_list), weights_list
