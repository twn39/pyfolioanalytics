# 进阶教程 3：滚动回测与交易成本 (Walk-Forward Backtesting)

在前两篇教程中，我们计算的都是 **“静态的最优解”**：我们在期初获取了过去几年的数据，计算出一个最优权重，然后就结束了。

但在现实交易中，随着时间的推移，资产的价格和波动率每天都在发生变化。昨天的“最优组合”，在下个月可能就变成了“高危组合”。因此，任何合格的量化策略在实盘部署前，都必须经过时间切片机的考验——**Walk-Forward 滚动回测**。

本教程将带你实现一个每月动态调仓、并扣除真实交易手续费的完整回测工作流。

## 1. 什么是 Walk-Forward 滚动回测？

假设我们的全量数据有 3 年：
1. **时间窗口**：我们定义一个为期 1 年（约 252 个交易日）的 **“训练窗口 (rolling_window)”**。
2. **第一期**：用第 1 天到第 252 天的数据计算历史参数，并跑出当月的最优权重。
3. **推进与持仓**：按照这个权重在真实市场中持有 1 个月，记录下这 1 个月每天的真实收益和账户净值。
4. **重新平衡**：1 个月后，训练窗口整体向右平移 1 个月（获取了最新的数据，丢弃了最老的数据），重新计算最优权重，并产生调仓换手。
5. 重复步骤 3-4，直到走到数据的尽头。

`PyFolioAnalytics` 内置了强大的 `backtest_portfolio` 引擎来自动处理这一切，甚至连复杂的**权重漂移 (Weight Drift)**（即因为涨跌导致的被动仓位偏离）都帮你计算在内。

## 2. 定义我们的策略

我们将继续使用在教程 2 中定义好的、极具实战意义的**“受合规约束的最大夏普比率策略”**：

```python
import pandas as pd
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.backtest import backtest_portfolio
from pyfolioanalytics.plots import plot_performance, plot_weights

# 1. 读取数据
returns = pd.read_csv("data/stock_returns.csv", index_col="Date", parse_dates=True)
port = Portfolio(assets=list(returns.columns))

# 2. 声明约束：满仓、只做多、个股 5%-30%、互联网板块最多 50%
port.add_constraint(type="weight_sum", min_sum=1.0, max_sum=1.0)
port.add_constraint(type="long_only")
port.add_constraint(type="box", min=0.05, max=0.30)
port.add_constraint(
    type="group", 
    groups=[[0, 4], [1, 2, 3]], # 0和4为硬件，1/2/3为互联网
    group_min=[0.0, 0.0], 
    group_max=[1.0, 0.5]
)

# 3. 声明目标：最大夏普 (收益/波动率)
port.add_objective(type="return", name="mean")
port.add_objective(type="risk", name="StdDev")
```

## 3. 启动回测引擎

接下来是核心代码：我们告诉回测引擎按“月”进行调仓，并且设置每次买卖需要缴纳 0.1% 的手续费 (PTC, Proportional Transaction Costs)。

```python
print("正在执行滚动回测 (Walk-Forward Backtest)...")

# 执行回测
bt = backtest_portfolio(
    R=returns, 
    portfolio=port, 
    optimize_method="ROI",
    rebalance_periods="ME",  # "ME" 代表 Month End (每月末调仓)
    rolling_window=252,      # 使用过去 252 个交易日的数据估计参数
    max_ratio=True,          # 我们同时有收益和风险目标，开启夏普比率最大化
    ptc=0.001                # 比例交易成本 (Proportional Transaction Costs) 设为 0.1%
)

print(f"\n回测完成！总期数: {len(bt.returns)}")
```

## 4. 分析回测结果

`backtest_portfolio` 运行结束后，会返回一个结构化的 `BacktestResult` 对象，里面记录了整个时间轴上每一天的动态！

我们可以直接获取它的复合收益率和换手率数据：

```python
# bt.net_returns 是一维的 pd.Series，记录了扣除交易费后每天的真实策略净收益
total_return = (1 + bt.net_returns).prod() - 1

# bt.turnover 记录了每天因为主动调仓和被动漂移产生的换手率
annual_turnover = bt.turnover.mean() * 252

print(f"年化换手率 (估算): {annual_turnover:.2%}")
print(f"扣除手续费后的总复合收益率: {total_return:.2%}")
```

在我们的实际 FAAMG 数据（2023-2026）下，由于大盘科技股表现强劲，你将得到类似如下的惊人表现：

```text
回测完成！总期数: 741
年化换手率 (估算): 71.62%
扣除手续费后的总复合收益率: 86.57%
```
*(注：这说明我们的策略在每月月末都会微调约 ~6% 的仓位，全年换手约 71%，在扣除这部分的印花税和佣金后，依旧在持有期内达成了 86% 的总收益。)*

## 5. 可视化图表 (Plotting)

量化投资永远离不开可视化。`PyFolioAnalytics` 提供了开箱即用的专业绘图函数，底层基于交互式的 `Plotly`。

你只需要将 `bt` 对象中的数据传给绘图函数，就能在浏览器中打开专业的交互式分析图表：

```python
# 图表 1：绘制随时间变化的策略净值曲线，以及底部的水下回撤 (Underwater Drawdown)
fig_perf = plot_performance(bt.net_returns, title="策略历史净值与回撤 (扣除手续费)")
fig_perf.show()

# 图表 2：绘制极其优雅的动态仓位面积图 (Area Chart)
# 这能让你直观看到：某个月份是否发生了大面积的板块轮动？
fig_weights = plot_weights(bt.weights, title="策略历史动态仓位分布")
fig_weights.show()
```

通过这些图表，你不再是面对冷冰冰的数字，而是能直观地向你的客户或基金经理展示这套风控组合在历史长河中是如何抵御黑天鹅、稳定积累净值的！

---

## 总结

恭喜你！到这里为止，你已经掌握了工业级量化投研从 0 到 1 的完整标准生命周期：
1. **数据输入** (`R`)
2. **约束与目标定义** (`Portfolio`, `add_objective`, `add_constraint`)
3. **滚动回测与成本** (`backtest_portfolio`, `ptc`)
4. **可视化与归因** (`plot_performance`)

在下一篇教程中，我们将步入深水区：如何对抗充满噪音的历史数据？敬请期待 **教程 4：参数降噪与随机矩阵理论 (RMT)**！
