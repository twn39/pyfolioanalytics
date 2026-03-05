from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd

class Portfolio:
    def __init__(
        self,
        assets: Union[int, List[str], Dict[str, float]],
        name: str = "portfolio",
        category_labels: Optional[Dict[str, List[int]]] = None,
        weight_seq: Optional[np.ndarray] = None,
    ):
        self.name = name
        self.weight_seq = weight_seq
        self.constraints = []
        self.objectives = []
        if isinstance(assets, int):
            self.assets = {f"Asset.{i+1}": 1.0 / assets for i in range(assets)}
        elif isinstance(assets, list):
            self.assets = {asset: 1.0 / len(assets) for asset in assets}
        elif isinstance(assets, dict):
            self.assets = assets
        else:
            raise ValueError("assets must be int, list, or dict")
        self.category_labels = category_labels

    def add_constraint(self, type: str, enabled: bool = True, **kwargs):
        nassets = len(self.assets)
        asset_names = list(self.assets.keys())
        constraint = {"type": type, "enabled": enabled}
        if type in ["weight_sum", "leverage", "full_investment", "dollar_neutral", "active"]:
            min_sum, max_sum = 0.99, 1.01
            if type == "full_investment": min_sum, max_sum = 1.0, 1.0
            elif type in ["dollar_neutral", "active"]: min_sum, max_sum = 0.0, 0.0
            else:
                min_sum = kwargs.get("min_sum", min_sum)
                max_sum = kwargs.get("max_sum", max_sum)
            constraint.update({"min_sum": min_sum, "max_sum": max_sum})
        elif type in ["box", "long_only"]:
            min_val, max_val = 0.0, 1.0
            if type == "box":
                min_val = kwargs.get("min", 0.0)
                max_val = kwargs.get("max", 1.0)
            if isinstance(min_val, (int, float)): min_vec = np.full(nassets, float(min_val))
            else: min_vec = np.array(min_val)
            if isinstance(max_val, (int, float)): max_vec = np.full(nassets, float(max_val))
            else: max_vec = np.array(max_val)
            constraint.update({"min": pd.Series(min_vec, index=asset_names), "max": pd.Series(max_vec, index=asset_names)})
        elif type == "group":
            constraint.update({"groups": kwargs.get("groups"), "group_min": kwargs.get("group_min"), "group_max": kwargs.get("group_max")})
        elif type in ["turnover", "transaction_cost"]:
            init_weights = kwargs.get("weight_initial")
            if init_weights is None: init_weights = np.array(list(self.assets.values()))
            elif isinstance(init_weights, dict): init_weights = np.array([init_weights.get(name, 0.0) for name in asset_names])
            elif isinstance(init_weights, pd.Series): init_weights = init_weights.reindex(asset_names).values
            constraint.update({"weight_initial": init_weights})
            if type == "turnover": constraint.update({"turnover_target": kwargs.get("turnover_target")})
            else:
                ptc = kwargs.get("ptc", 0.0)
                if isinstance(ptc, (int, float)): ptc = np.full(nassets, float(ptc))
                constraint.update({"ptc": ptc})
        elif type == "position_limit":
            constraint.update({"max_pos": kwargs.get("max_pos"), "max_pos_long": kwargs.get("max_pos_long"), "max_pos_short": kwargs.get("max_pos_short")})
        else:
            constraint.update(kwargs)
        self.constraints.append(constraint)
        return self

    def add_objective(self, type: str, name: Optional[str] = None, enabled: bool = True, arguments: Optional[Dict[str, Any]] = None, multiplier: Optional[float] = None, **kwargs):
        if type == "quadratic_utility":
            risk_aversion = kwargs.get("risk_aversion", 1.0)
            self.add_objective(type="return", name="mean", enabled=enabled)
            self.add_objective(type="risk", name="var", enabled=enabled, risk_aversion=risk_aversion)
            return self
        if multiplier is None:
            if type in ["return", "return_objective"]: multiplier = -1.0
            else: multiplier = 1.0
        obj = {"type": type, "name": name, "enabled": enabled, "arguments": arguments or {}, "multiplier": multiplier, "target": kwargs.get("target")}
        obj.update(kwargs)
        self.objectives.append(obj)
        return self

    def clear_objectives(self):
        self.objectives = []
        return self

    def get_constraints(self) -> Dict[str, Any]:
        nassets = len(self.assets)
        asset_names = list(self.assets.keys())
        out = {
            "min_sum": 1.0, "max_sum": 1.0,
            "min": pd.Series(np.full(nassets, -np.inf), index=asset_names),
            "max": pd.Series(np.full(nassets, np.inf), index=asset_names),
            "groups": None, "turnover_target": None, "ptc": None, "weight_initial": None,
            "max_pos": None, "max_pos_long": None, "max_pos_short": None,
            "min_return": None
        }
        for constr in self.constraints:
            if not constr.get("enabled", True): continue
            ctype = constr["type"]
            if ctype in ["weight_sum", "leverage", "full_investment", "dollar_neutral", "active"]:
                out["min_sum"] = constr["min_sum"]
                out["max_sum"] = constr["max_sum"]
            elif ctype in ["box", "long_only"]:
                out["min"] = constr["min"]
                out["max"] = constr["max"]
            elif ctype == "group":
                out["groups"] = constr["groups"]
                out["group_min"] = constr["group_min"]
                out["group_max"] = constr["group_max"]
            elif ctype == "turnover":
                out["turnover_target"] = constr["turnover_target"]
                out["weight_initial"] = constr["weight_initial"]
            elif ctype == "transaction_cost":
                out["ptc"] = constr["ptc"]
                out["weight_initial"] = constr["weight_initial"]
            elif ctype == "position_limit":
                out["max_pos"] = constr.get("max_pos")
                out["max_pos_long"] = constr.get("max_pos_long")
                out["max_pos_short"] = constr.get("max_pos_short")
            elif ctype == "return":
                out["min_return"] = constr.get("min_return")
        return out

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def __repr__(self):
        return f"Portfolio(name={self.name}, assets={list(self.assets.keys())})"

class RegimePortfolio:
    """
    Class to hold multiple portfolios corresponding to different market regimes.
    """
    def __init__(self, portfolios: List[Portfolio], regime_labels: List[Any]):
        if len(portfolios) != len(regime_labels):
            raise ValueError("Number of portfolios must match number of regime labels.")
        self.portfolios = dict(zip(regime_labels, portfolios))
        self.regime_labels = regime_labels

    def get_portfolio(self, regime: Any) -> Portfolio:
        if regime not in self.portfolios:
            # Fallback to the first one if regime not found?
            return next(iter(self.portfolios.values()))
        return self.portfolios[regime]

    def __repr__(self):
        return f"RegimePortfolio(regimes={self.regime_labels})"

class MultLayerPortfolio:
    """
    Class to hold a hierarchical portfolio structure.
    """
    def __init__(self, root_portfolio: Portfolio):
        self.root = root_portfolio
        self.sub_portfolios = {} # Map meta-asset name to Portfolio

    def add_sub_portfolio(self, meta_asset_name: str, sub_portfolio: Portfolio):
        if meta_asset_name not in self.root.assets:
            raise ValueError(f"'{meta_asset_name}' must be defined as an asset in the root portfolio.")
        self.sub_portfolios[meta_asset_name] = sub_portfolio

    def __repr__(self):
        return f"MultLayerPortfolio(root={self.root.name}, sub={list(self.sub_portfolios.keys())})"
