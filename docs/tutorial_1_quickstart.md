# 快速入门：真实世界的科技股投资组合

欢迎使用 PyFolioAnalytics！由于量化组合优化涉及较多数学概念，直接阅读 API 文档可能会让人感到困惑。本篇教程将通过一个 **真实世界的科技股投资组合（FAAMG）** 案例，由浅入深地带你体验 PyFolioAnalytics 的核心工作流。

这是我们的入门教程，目标是构建一个 **“最低波动率（最小方差）”** 的多头投资组合。

## 1. 准备工作与数据读取

在真实世界的量化投资中，第一步总是获取资产的收益率数据。我们在 `data/stock_returns.csv` 中为你准备了五只美国科技巨头（苹果 AAPL、亚马逊 AMZN、谷歌 GOOGL、Meta META、微软 MSFT）过去三年（2023-04-15 至 2026-04-15）的真实日度收益率数据。

首先，导入我们需要的库，并读取数据：

```python
import pandas as pd
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

# 1. 读取历史日度收益率数据
# 注意：在量化分析中，时间序列的索引必须是日期格式
returns = pd.read_csv("data/stock_returns.csv", index_col="Date", parse_dates=True)

# 查看前几行数据
print(returns.head())
```

## 2. 声明投资组合 (Portfolio Specification)

在 PyFolioAnalytics 中，我们采用**声明式**的设计理念。你不需要自己去写复杂的优化目标数学公式（如二次规划），只需创建一个 `Portfolio` 对象，并告诉它你的规则。

```python
# 获取资产名称列表 (AAPL, AMZN, GOOGL, META, MSFT)
assets = list(returns.columns)

# 2. 初始化投资组合
port = Portfolio(assets=assets)
```

接下来，给我们的投资组合添加一些“现实世界”的约束条件：

```python
# 约束 1：满仓（所有资产权重之和等于 100%）
port.add_constraint(type="weight_sum", min_sum=1.0, max_sum=1.0)

# 约束 2：只能做多（权重不能为负，不允许做空）
port.add_constraint(type="long_only")
```

有了约束，我们还需要一个优化的目标。在这个例子中，我们是一个保守的投资者，目标是**最小化投资组合的整体风险（波动率）**。

```python
# 目标：最小化风险，风险指标选择标准差 (StdDev / Variance)
port.add_objective(type="risk", name="StdDev")
```

```python
# 3. 求解最优配置
# optimize_portfolio 在底层会自动帮你计算历史协方差矩阵 (moments) 并调用凸优化引擎 (ROI)
res = optimize_portfolio(
    R=returns, 
    portfolio=port, 
    optimize_method="ROI"
)

# 检查求解状态
print(f"优化状态: {res['status']}")

# 输出最终的资产最优权重
if res['status'] == "optimal":
    print("最优权重分布:")
    for asset, weight in res['weights'].items():
        print(f"{asset}: {weight:.2%}")
```

## 4. 预期输出与解释

运行上述代码，你将会得到类似以下的输出：

```text
优化状态: optimal
最优权重分布:
AAPL: 33.73%
AMZN: 0.02%
GOOGL: 18.55%
META: 0.00%
MSFT: 47.70%
```
*(注：具体数值取决于你所用时间段内的历史数据区间，上述为示例展示)*

**解读：**
1. **满仓且做多**：所有权重加起来等于 100%，且没有负数。
2. **避开高波动**：在过去三年的行情中，META 与 AMZN 的走势波动相对较大且与组合其他成分股具有不同程度的相关性，因此被优化引擎判定为“不划算的风险”，分配了极低（0.02% 或 0%）的权重。而相对稳健的微软（MSFT）和苹果（AAPL）获得了绝大部分权重（合计超过 80%）以最小化整体投资组合的方差。

## 下一步

这只是一个最基础的“最小方差组合”！在现实的投资机构中，你可能会面临更复杂的挑战：
- **最大化收益风险比**（如夏普比率）。
- 某只股票由于合规要求，**最大持仓不能超过 15%**，或者某几个板块必须在规定比例内。
- 历史协方差矩阵充满噪音，需要使用 **随机矩阵理论 (RMT)** 进行降噪处理。
- 投资者不仅关注标准差，更害怕尾部黑天鹅风险，需要最小化 **CVaR (条件在险价值)** 或最大回撤。

在后续的进阶教程中，我们将逐步引入这些高级特性，带你构建工业级的量化投资组合。
