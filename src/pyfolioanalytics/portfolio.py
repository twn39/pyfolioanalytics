from typing import List, Dict, Any, Union, Optional
import numpy as np
import pandas as pd


class Portfolio:
    def __init__(
        self,
        assets: Union[int, List[str], Dict[str, float]],
        name: str = "portfolio",
    ):
        self.name = name
        if isinstance(assets, int):
            self.assets = {f"Asset.{i + 1}": 0.0 for i in range(assets)}
        elif isinstance(assets, list):
            self.assets = {a: 0.0 for a in assets}
        else:
            self.assets = assets
        self.constraints = []
        self.objectives = []

    def add_constraint(self, type: str, enabled: bool = True, **kwargs):
        constraint = {"type": type, "enabled": enabled}
        nassets = len(self.assets)
        asset_names = list(self.assets.keys())

        if type in [
            "weight_sum",
            "leverage",
            "full_investment",
            "dollar_neutral",
            "active",
        ]:
            min_sum, max_sum = 0.99, 1.01
            if type == "full_investment":
                min_sum, max_sum = 1.0, 1.0
            elif type in ["dollar_neutral", "active"]:
                min_sum, max_sum = 0.0, 0.0
            else:
                min_sum = kwargs.get("min_sum", min_sum)
                max_sum = kwargs.get("max_sum", max_sum)
            constraint.update({"min_sum": min_sum, "max_sum": max_sum})
        elif type in ["box", "long_only"]:
            if type == "long_only":
                min_val, max_val = 0.0, 1.0
            else:
                min_val = kwargs.get("min", 0.0)
                max_val = kwargs.get("max", 1.0)

            if isinstance(min_val, (int, float)):
                min_vec = np.full(nassets, float(min_val))
            else:
                min_vec = np.array(min_val)

            if isinstance(max_val, (int, float)):
                max_vec = np.full(nassets, float(max_val))
            else:
                max_vec = np.array(max_val)

            constraint.update(
                {
                    "min": pd.Series(min_vec, index=asset_names),
                    "max": pd.Series(max_vec, index=asset_names),
                }
            )
        elif type == "group":
            constraint.update(
                {
                    "groups": kwargs.get("groups"),
                    "group_min": kwargs.get("group_min"),
                    "group_max": kwargs.get("group_max"),
                }
            )
        elif type in ["turnover", "transaction_cost"]:
            init_weights = kwargs.get("weight_initial")
            if init_weights is None:
                init_weights = np.array(list(self.assets.values()))
            elif isinstance(init_weights, dict):
                init_weights = np.array(
                    [init_weights.get(name, 0.0) for name in asset_names]
                )
            elif isinstance(init_weights, pd.Series):
                init_weights = init_weights.reindex(asset_names).values
            constraint.update({"weight_initial": init_weights})
            if type == "turnover":
                constraint.update({"turnover_target": kwargs.get("turnover_target")})
            else:
                ptc = kwargs.get("ptc", 0.0)
                if isinstance(ptc, (int, float)):
                    ptc = np.full(nassets, float(ptc))
                constraint.update({"ptc": ptc})
        elif type == "position_limit":
            constraint.update({"max_pos": kwargs.get("max_pos")})
        elif type == "tracking_error":
            constraint.update(
                {
                    "target": kwargs.get("target", 0.05),
                    "benchmark": kwargs.get("benchmark"),
                }
            )
        elif type == "active_share":
            constraint.update(
                {
                    "active_share_target": kwargs.get("target", 0.6),
                    "active_share_benchmark": kwargs.get("benchmark"),
                }
            )
        elif type == "factor_exposure":
            B = kwargs.get("B")
            lower = kwargs.get("lower")
            upper = kwargs.get("upper")
            
            if B is None or lower is None or upper is None:
                raise ValueError("Factor exposure constraint requires B, lower, and upper.")
            
            # Convert B to matrix if it's a list or vector
            if isinstance(B, (list, np.ndarray)) and not isinstance(B, pd.DataFrame):
                B = np.array(B)
                if B.ndim == 1:
                    B = B.reshape(-1, 1)
            
            # Validate dimensions
            if B.shape[0] != nassets:
                raise ValueError(f"B must have {nassets} rows (number of assets).")
            
            nfactors = B.shape[1]
            if len(lower) != nfactors or len(upper) != nfactors:
                raise ValueError(f"lower and upper must have length {nfactors} (number of factors).")
            
            constraint.update({
                "B": B,
                "lower": np.array(lower),
                "upper": np.array(upper)
            })
        elif type == "robust":
            delta_mu = kwargs.get("delta_mu", 0.0)
            if isinstance(delta_mu, (int, float)):
                delta_mu_vec = np.full(nassets, float(delta_mu))
            else:
                delta_mu_vec = np.array(delta_mu)
            constraint.update({"delta_mu": pd.Series(delta_mu_vec, index=asset_names)})

        self.constraints.append(constraint)
        return self

    def add_objective(
        self,
        type: str,
        name: Optional[str] = None,
        enabled: bool = True,
        arguments: Optional[Dict[str, Any]] = None,
        multiplier: Optional[float] = None,
        **kwargs,
    ):
        if name is None:
            name = type
        if multiplier is None:
            if type in ["return", "return_objective"]:
                multiplier = -1.0
            else:
                multiplier = 1.0
        obj = {
            "type": type,
            "name": name,
            "enabled": enabled,
            "arguments": arguments or {},
            "multiplier": multiplier,
            "target": kwargs.get("target"),
        }
        obj.update(kwargs)
        self.objectives.append(obj)
        return self

    def get_constraints(self) -> Dict[str, Any]:
        asset_names = list(self.assets.keys())
        len(asset_names)
        res = {
            "min_sum": -np.inf,
            "max_sum": np.inf,
            "min": pd.Series(-np.inf, index=asset_names),
            "max": pd.Series(np.inf, index=asset_names),
        }
        for constr in self.constraints:
            if not constr.get("enabled", True):
                continue
            ctype = constr["type"]
            if ctype in [
                "weight_sum",
                "leverage",
                "full_investment",
                "dollar_neutral",
                "active",
            ]:
                res["min_sum"] = max(res["min_sum"], constr["min_sum"])
                res["max_sum"] = min(res["max_sum"], constr["max_sum"])
            elif ctype in ["box", "long_only"]:
                res["min"] = np.maximum(res["min"], constr["min"])
                res["max"] = np.minimum(res["max"], constr["max"])
            else:
                res.update(
                    {k: v for k, v in constr.items() if k not in ["type", "enabled"]}
                )

        # Final cleanup of box defaults if still -inf/inf
        res["min"] = res["min"].replace(-np.inf, -1.0)
        res["max"] = res["max"].replace(np.inf, 1.0)
        return res

    def clear_objectives(self):
        self.objectives = []
        return self

    def copy(self):
        import copy

        return copy.deepcopy(self)

    def __repr__(self):
        return f"Portfolio(name={self.name}, assets={list(self.assets.keys())})"


class RegimePortfolio:
    def __init__(self, portfolios: List[Portfolio], regime_labels: List[Any]):
        if len(portfolios) != len(regime_labels):
            raise ValueError("Number of portfolios must match number of regime labels.")
        self.portfolios = dict(zip(regime_labels, portfolios))
        self.regime_labels = regime_labels

    def get_portfolio(self, regime: Any) -> Portfolio:
        if regime not in self.portfolios:
            return next(iter(self.portfolios.values()))
        return self.portfolios[regime]

    def clear_objectives(self):
        self.objectives = []
        return self

    def copy(self):
        import copy

        return copy.deepcopy(self)

    def __repr__(self):
        return f"RegimePortfolio(regimes={self.regime_labels})"


class MultLayerPortfolio:
    def __init__(self, root_portfolio: Portfolio):
        self.root = root_portfolio
        self.sub_portfolios = {}

    def add_sub_portfolio(self, meta_asset_name: str, sub_portfolio: Portfolio):
        if meta_asset_name not in self.root.assets:
            raise ValueError(
                f"'{meta_asset_name}' must be defined as an asset in the root portfolio."
            )
        self.sub_portfolios[meta_asset_name] = sub_portfolio

    def clear_objectives(self):
        self.objectives = []
        return self

    def copy(self):
        import copy

        return copy.deepcopy(self)

    def __repr__(self):
        return f"MultLayerPortfolio(root={self.root.name}, sub={list(self.sub_portfolios.keys())})"
