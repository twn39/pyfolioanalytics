# 回测与可视化 (`backtest.py` & `plots.py`)

本模块提供投资组合随时间再平衡和滚动窗口验证的基础引擎，并集成了基于 `plotly` 的精美可视化报表组件。

## 回测 API (`backtest.py`)

### `optimize_portfolio_rebalancing(R, portfolio, ...)`
用于执行带有滚动窗口、定期调仓配置约束的带时间感知的策略回测。

- **核心参数**:
  - `R` (pd.DataFrame): 完整的历史收益序列。
  - `portfolio` (Portfolio): 定义好的组合约束或多周期 Regime 规则。
  - `training_period` (int): 初始训练集的长度（滚动窗口大小）。
  - `rolling_window` (int, 可选): 如果不为空，将在每个再平衡期使用定长窗口数据，否则使用不断扩展的窗口。
  - `rebalance_on` (str): 重新优化的触发周期。例如 `'months'`, `'quarters'`, `'years'`。
  - `optimize_method` (str): 求解器配置。

- **返回**: 
  - `BacktestResult` 实例对象。可通过 `result.extract_weights()` 取出时间序列权重矩阵，通过 `result.extract_returns()` 取出回测期组合的净值收益率序列。

## 可视化 API (`plots.py`)

所有的绘图函数均返回交互式的 `plotly.graph_objects.Figure` 对象。

- **`plot_weights(weights_df)`**:
  传入 `T \times N` 的权重变化 DataFrame，绘制堆叠面积图，直观展现资产占比的历史演变及换手情况。

- **`plot_performance(returns, benchmark=None)`**:
  绘制投资组合随时间复利增长的净值曲线 (Cumulative Return Curve)。

- **`plot_efficient_frontier(R, portfolio, ...)`**:
  接受收益数据与 Portfolio 对象，自动扫描风险与收益维度的边界范围，并绘制抛物线状的**有效前沿 (Efficient Frontier)**，以及组成有效前沿的离散最优组合节点散点。

- **`plot_risk_decomposition(weights, sigma, risk_type="variance")`**:
  条形图呈现组合中每个成分资产对应的风险边际贡献百分比 (Percentage Marginal Risk Contribution)，是评估风险平价达成情况的重要报表。

## 代码示例

```python
import pandas as pd
import numpy as np
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.backtest import optimize_portfolio_rebalancing
from pyfolioanalytics.plots import plot_weights, plot_performance

# 模拟 3 年的日度数据 (约 750 个交易日)
dates = pd.date_range('2021-01-01', periods=750, freq='B')
R = pd.DataFrame(np.random.randn(750, 4) * 0.01 + 0.0005, index=dates, columns=['A', 'B', 'C', 'D'])

port = Portfolio(assets=list(R.columns))
port.add_constraint(type="long_only").add_constraint(type="full_investment")
port.add_objective(type="risk", name="StdDev")

# 每月 (约 21 交易日) 根据过去 100 天的数据重新平衡
bt_result = optimize_portfolio_rebalancing(
    R, port, 
    optimize_method="ROI",
    training_period=100,
    rolling_window=100,
    rebalance_on="months"
)

# 提取并画图
weights_history = bt_result.extract_weights()
returns_history = bt_result.extract_returns()

fig_w = plot_weights(weights_history)
# fig_w.show() # 在 Jupyter Notebook 渲染交互图表

fig_p = plot_performance(returns_history)
# fig_p.show()
```