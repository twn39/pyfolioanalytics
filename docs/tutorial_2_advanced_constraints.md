# 进阶教程 2：追求夏普比率与现实合规约束

在第一个教程中，我们构建了一个 **“最小方差组合”**，它的目标非常明确：把风险降到最低。在那里的结果中，保守的苹果和微软占据了绝大部分仓位。

但在现实投资中，我们不仅厌恶风险，我们也渴望利润。现代投资组合理论（MPT）的巅峰是寻找**“最大夏普比率（Sharpe Ratio）”**，即收益与风险的比值最大化（也称切线组合）。

本篇教程将带你从单纯的“躲避风险”，走向更成熟的“风险预算与合规约束”的工业级投资策略。

## 1. 为什么我们需要合规约束？

在讲解代码前，我们先看一个**反面教材**。如果我们在 PyFolioAnalytics 中，直接告诉机器：“给我最大化夏普比率（高收益 + 低波动）”，机器会在过去三年的数据中发现 GOOGL 和 META 的收益极其丰厚，从而给出这样偏激的无脑配置：

```text
# (反面教材) 无约束下的最大夏普比率权重：
AAPL: 0.00%
AMZN: 6.14%
GOOGL: 65.13%
META: 28.73%
MSFT: 0.00%
```

把高达 65% 的资金重仓在一只股票上？这在任何正规的公募基金或对冲基金里都是**绝对禁止**的！这不仅导致了单点风险过载，也违背了投资组合分散化的初衷。

因此，现实业务中，基金经理必须面临大量 **合规与风控约束（Constraints）**。

## 2. 声明复杂的业务规则

我们将引入两种最常用的业务约束：**个股比例上限 (Box)** 和 **板块行业上限 (Group)**。

假设我们这只科技主题基金的合规手册写明：
1. **底仓与限购**：任何一只入池股票，最低仓位不得低于 5%，最高不得超过 30%。
2. **行业平衡**：我们将五大巨头分为两组：
   - 硬件与系统：苹果 (AAPL)、微软 (MSFT)
   - 互联网服务：亚马逊 (AMZN)、谷歌 (GOOGL)、Meta (META)
   - 要求：互联网服务板块的波动太大，其总仓位加起来 **不能超过 50%**。

在 `PyFolioAnalytics` 中，实现这些复杂的业务逻辑不需要写一页纸的 if-else，只需声明约束即可：

```python
import pandas as pd
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

# 1. 读取数据
returns = pd.read_csv("data/stock_returns.csv", index_col="Date", parse_dates=True)
assets = list(returns.columns)

port = Portfolio(assets=assets)

# 基础约束：满仓、只做多
port.add_constraint(type="weight_sum", min_sum=1.0, max_sum=1.0)
port.add_constraint(type="long_only")

# 进阶约束 1：个股上限与下限 (Box Constraint)
port.add_constraint(type="box", min=0.05, max=0.30)

# 进阶约束 2：板块组约束 (Group Constraint)
# 按列索引分组: 硬件 [0, 4], 互联网 [1, 2, 3]
port.add_constraint(
    type="group", 
    groups=[[0, 4], [1, 2, 3]], 
    group_min=[0.0, 0.0], 
    group_max=[1.0, 0.5] # 第二个元素 0.5 代表互联网组最高 50%
)
```

## 3. 设置多重目标与夏普比率求解

现在规则定好了，我们要告诉优化器我们的目标：既要高收益，又要低波动。

```python
# 目标 1：最大化期望收益
port.add_objective(type="return", name="mean")

# 目标 2：最小化波动率
port.add_objective(type="risk", name="StdDev")

# 求解优化
# 当同时存在 return 和 risk 两个目标时，开启 max_ratio=True
# 求解器底层会自动使用 Charnes-Cooper 变换，将其转换为求解最大夏普比率问题
print("正在调用凸优化引擎进行多约束夏普比率最大化...")
res = optimize_portfolio(
    R=returns, 
    portfolio=port, 
    optimize_method="ROI",
    max_ratio=True  # 核心参数：求比值最大化
)

print(f"优化状态: {res['status']}")
if res['status'] == "optimal":
    for asset, weight in res['weights'].items():
        print(f"{asset}: {weight:.2%}")
```

## 4. 结果的数学之美与业务解读

运行上述完整代码，底层强大的 CVXPY 凸优化引擎会在多维空间中寻找平衡点，最终你会得到如下输出：

```text
优化状态: optimal
AAPL: 30.00%
AMZN: 5.00%
GOOGL: 30.00%
META: 15.00%
MSFT: 20.00%
```

**这是一组令人拍案叫绝的配置结果，深刻展现了数学引擎是如何完美遵守业务指令的：**
1. **逼近天花板**：谷歌 (GOOGL) 收益贡献大，引擎想多买，但由于我们设置了 `box` 个股最高 30% 约束，它精准地停在了 **30.00%**。苹果 (AAPL) 同理，顶格买入 **30.00%**。
2. **守住底线**：亚马逊 (AMZN) 在这三年数据中相对较弱，引擎本想清仓（0%），但因为我们有“底仓不低于 5%”的规定，它被精准按在了 **5.00%**。
3. **行业板块的博弈**：最精彩的部分来了！为什么 META 是 15%？
   - 我们的“互联网板块”规定最高只能买 50%。
   - 引擎为了最大化夏普，已经给 GOOGL 分配了 30% (打满上限)，AMZN 被强制分配了 5% (底仓下限)。
   - 互联网板块的可用额度只剩下： `50% - 30% - 5% = 15%`。
   - 因此，META 被不多不少、极为精准地分配了 **15.00%** 的剩余全额额度！
4. **填补缺口**：最后，为了满足总仓位 100% 的要求，剩下的 20.00% 份额顺理成章地给到了硬件组的微软 (MSFT)。

## 总结

通过这篇教程，你掌握了如何在 `PyFolioAnalytics` 中使用 `max_ratio=True` 求解切线组合（最大夏普），并且见识到了 `box`（个股约束）和 `group`（板块约束）在真实业务中的威力。

不管你给出多么复杂、交织的合规要求，我们的凸优化底层都能帮你找到那个完美符合所有条件、且收益风险比最高的黄金配置点。
