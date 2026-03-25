# 组合规格定义 (`portfolio.py`)

本模块的核心是 `Portfolio` 类。它采用声明式编程模型，允许用户在不触及任何底层数学求解器代码的情况下，灵活叠加各种业务规则与目标。

## 功能描述

`Portfolio` 对象作为参数传递的载体，负责存储：
- 资产池清单。
- **约束条件 (Constraints)**：例如权重上下限、多空限制、杠杆比例、因子敞口、集中度限制、自定义线性矩阵约束等。
- **优化目标 (Objectives)**：例如最大化收益、最小化风险、风险预算（风险平价）等。

## 核心 API

### `Portfolio(assets)`
- **输入**: 
  - `assets` (List[str] | Dict): 资产名称列表，或带有初始权重的字典。
- **输出**: 初始化后的 Portfolio 实例。

### `add_constraint(type, enabled=True, **kwargs)`
添加一项组合约束。
- **输入**:
  - `type` (str): 约束类型。支持：
    - `"full_investment"`: 权重和为 1。
    - `"long_only"`: 权重非负。
    - `"box"`: 接受 `min` / `max` 列表或浮点数，控制单资产权重区间。
    - `"weight_sum"`: 接受 `min_sum` / `max_sum`。
    - `"linear"`: 自定义线性约束，接受 `A`, `b`, `A_eq`, `b_eq` 参数（对应 $Aw \le b$）。
    - `"turnover"`, `"transaction_cost"`, `"factor_exposure"`, `"active_share"` 等。
- **输出**: `self` (支持链式调用)。

### `add_objective(type, name, enabled=True, arguments=None, multiplier=1.0)`
添加一项优化目标。
- **输入**:
  - `type` (str): 目标分类。主要为 `"return"` (收益), `"risk"` (风险), `"risk_budget"` (风险预算/平价)。
  - `name` (str): 具体目标的名称。例如 `"mean"`, `"StdDev"`, `"VaR"`, `"MAD"`, `"CVaR"`, `"EDaR"`。
  - `arguments` (dict): 传递给风险计算函数的参数，如 `{"p": 0.95}` 代表 95% 置信度。
  - `multiplier` (float): 目标乘数（例如设置为 -1.0 可将最小化转为最大化）。
- **输出**: `self` (支持链式调用)。

## 代码示例

```python
import pandas as pd
import numpy as np
from pyfolioanalytics.portfolio import Portfolio

# 假设有 4 个资产
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

# 初始化组合
port = Portfolio(assets=assets)

# 1. 叠加约束
port.add_constraint(type="full_investment") \
    .add_constraint(type="long_only") \
    .add_constraint(type="box", min=0.05, max=0.40) # 单个资产占比 5% ~ 40%

# 2. 叠加自定义线性约束 (例如前两个资产的总权重大于 30%)
# A @ w <= b  =>  -w[0] - w[1] <= -0.3
A_matrix = [[-1, -1, 0, 0]]
b_vector = [-0.3]
port.add_constraint(type="linear", A=A_matrix, b=b_vector)

# 3. 叠加目标
# 最大化收益，最小化半绝对偏差 (Semi-MAD)
port.add_objective(type="return", name="mean")
port.add_objective(type="risk", name="semi_MAD")

print(port.get_constraints())
```