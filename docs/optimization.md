# 优化引擎 (`optimize.py` & `solvers.py`)

优化引擎是 PyFolioAnalytics 的心脏，它解析声明式的 `Portfolio`，将其映射并注入到底层凸优化求解器 (基于 `CVXPY`) 或启发式求解器 (基于 `SciPy`) 中。

## 核心 API

### `optimize_portfolio(R, portfolio, optimize_method="ROI", moments=None, **kwargs)`
调度相应的数学规划模型，执行资产配置权重的求解。

- **输入**:
  - `R` (pd.DataFrame): 历史收益率数据。如果已传入预先计算的 `moments`，某些测度下 `R` 可传 `None`。
  - `portfolio` (Portfolio): 包含资产、约束和目标的组合规格定义。
  - `optimize_method` (str): 求解后端。
    - `"ROI"`: 使用全局数学凸规划解析 (调用 `CVXPY` 映射 LP/QP/SOCP 锥规划)。这是最高效、最精确的方法。
    - `"random"`: 随机投资组合生成法。
    - `"DEoptim"`: 差分进化启发式算法，用于处理非凸的目标或复杂的约束条件。
  - `moments` (Dict, 可选): 如果不提供，函数将自动调用 `set_portfolio_moments` 进行计算。

- **输出**: `Dict[str, Any]`
  - `"status"`: 求解状态 (`"optimal"`, `"infeasible"`, `"failed"` 等)。
  - `"weights"`: 最优权重 (`pd.Series`)。
  - `"objective_measures"`: 评估得到的最优组合在各个目标下的具体数值。
  - `"moments"`: 本次优化使用的预估参数。

## 支持的数学规划架构 (Solvers)
系统会自动侦测您在 `portfolio.add_objective()` 中声明的目标，并无缝切入对应的线性/二阶锥转换公式：
- **Mean-Variance (StdDev)**: 二次规划 (QP)。
- **MAD / Semi-MAD**: 线性规划 (LP)。
- **VaR / ES (CVaR)**: 使用 Rockafellar-Uryasev 线性辅助变量表达 (LP)。
- **EVaR / EDaR**: 熵风险，通过指数锥规划 (Exponential Cone Programming) 求解。
- **OWA (L-Moments)**: 有序加权平均风险，转换为基于排名的线性规划系统 (LP)。
- **Drawdown (MaxDrawdown, CDaR)**: 回撤水位线动态追踪辅助变量系统。
- **Risk Parity (ERC)**: 风险预算平价，自动切入 `solve_nonlinear` 或特定非线性求解路径。

## 代码示例

```python
import pandas as pd
import numpy as np
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

R = pd.DataFrame(np.random.randn(200, 4) * 0.01, columns=['A', 'B', 'C', 'D'])
port = Portfolio(assets=list(R.columns))
port.add_constraint(type="full_investment").add_constraint(type="long_only")

# 示例 1: 最小化平均绝对偏差 (MAD) 优化 (全线性规划，极其稳健)
port_mad = port.copy()
port_mad.add_objective(type="risk", name="MAD")
res_mad = optimize_portfolio(R, port_mad)
print("Min MAD Weights:\n", res_mad["weights"])

# 示例 2: 最大化夏普比率 (CVXPY 后端)
port_sharpe = port.copy()
port_sharpe.add_objective(type="return", name="mean")
port_sharpe.add_objective(type="risk", name="StdDev")
res_sharpe = optimize_portfolio(R, port_sharpe)

# 示例 3: 最小化 95% 熵条件回撤 (EDaR - Entropic Drawdown at Risk)
port_edar = port.copy()
port_edar.add_objective(type="risk", name="EDaR", arguments={"p": 0.95})
res_edar = optimize_portfolio(R, port_edar)
print("Min EDaR Status:", res_edar["status"])
```