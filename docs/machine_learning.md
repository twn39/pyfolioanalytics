# 进阶教程 6：放弃求逆，用机器学习重塑风险平价 (Hierarchical Risk Parity)

马科维茨的“均值-方差”模型赢得了诺贝尔奖，但在过去的几十年实盘中，基金经理们对它又爱又恨。

最大的问题是：传统的凸优化器对协方差矩阵太“敏感”了。只要历史数据稍有噪音，甚至两只股票恰巧在过去某段时间相关性达到了 0.99，传统的数学引擎就会在求逆矩阵时崩溃，并给出一个荒谬的极端“梭哈”配置（例如：做多 A 股票 1000%，做空 B 股票 900%）。

2016 年，金融机器学习领军人物 Marcos López de Prado 提出了 **HRP（层次风险平价）** 算法。它彻底**放弃了矩阵求逆**，而是引入了无监督的“层次聚类（Hierarchical Clustering）”。

本篇教程，我们将离开科技股，使用量化界极具代表性的 **EDHEC 对冲基金指数**（涵盖 13 种不同的对冲策略的月度收益），来看看 HRP 是如何碾压传统马科维茨的。

## 1. 传统马科维茨的“梭哈”绝症 (Corner Solutions)

我们先用经典的 `StdDev`（最小方差）去跑这 13 个对冲基金指数，看看传统数学引擎会给我们什么建议：

```python
import pandas as pd
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio
from pyfolioanalytics.plots import plot_dendrogram

# 1. 读取 EDHEC 的 13 个子策略指数月度数据
returns = pd.read_csv("data/edhec.csv", index_col="date", parse_dates=True, dayfirst=True).iloc[:100]
port = Portfolio(assets=list(returns.columns))

# [对立面] 传统的马科维茨全局最小方差组合 (GMV)
port_gmv = port.copy()
port_gmv.add_constraint(type="weight_sum", min_sum=1.0, max_sum=1.0)
port_gmv.add_constraint(type="long_only")
port_gmv.add_objective(type="risk", name="StdDev")

res_gmv = optimize_portfolio(returns, port_gmv, optimize_method="ROI")

# 打印权重大于 1% 的资产
for asset, w in res_gmv['weights'].items():
    if w > 0.01:
        print(f"{asset}: {w:.2%}")
```

**传统模型的结果（灾难性的集中度）**：
```text
Distressed Securities: 3.45%
Equity Market Neutral: 72.15%
Fixed Income Arbitrage: 3.87%
Relative Value: 14.66%
Short Selling: 5.86%
```
看到了吗？尽管我们输入了 13 种截然不同、用来分散风险的对冲策略，但优化器却把 **72.15%** 的资金全部“梭哈”给了 `Equity Market Neutral`（股票市场中性）这一个策略！剩下的 8 个策略（如全球宏观、新兴市场等）直接被分配了 **0权重**。

这不仅违背了我们“分散投资”的初衷，而且一旦 `Equity Market Neutral` 在未来遭遇黑天鹅，整个母基金将面临灭顶之灾。

## 2. 拥抱机器学习：调用 HRP

HRP 算法从根本上改变了游戏规则。它认为：**高度相关的资产应该被视为一个整体（Cluster）进行预算分配，而不是让它们互相厮杀抢夺权重。**

在 `PyFolioAnalytics` 中，调用 HRP 非常简单，因为它**不需要**声明具体的线性约束（它内置了基于树图的等权重切分和 `long_only` 满仓分配逻辑）：

```python
# [进化] 层次风险平价 (HRP - Hierarchical Risk Parity)
res_hrp = optimize_portfolio(
    returns, 
    port, 
    optimize_method="HRP", 
    linkage="single" # 聚类距离链的计算方式
)

# 打印所有资产权重
for asset, w in res_hrp['weights'].items():
    print(f"{asset}: {w:.2%}")
```

**HRP 惊艳的分配结果**：
```text
Convertible Arbitrage: 13.90%
CTA Global: 3.54%
Distressed Securities: 2.00%
Emerging Markets: 1.02%
Equity Market Neutral: 33.69%
Event Driven: 1.87%
Fixed Income Arbitrage: 20.42%
Global Macro: 4.27%
Long/Short Equity: 1.18%
Merger Arbitrage: 6.95%
Relative Value: 8.68%
Short Selling: 0.69%
Funds Of Funds: 1.78%
```

你看！HRP 没有丢下任何一个策略！
它并没有无脑地“平均分（每个 7.6%）”，也没有偏执地“把 72% 给一个人”。
它通过机器学习识别到 `Equity Market Neutral` 和 `Fixed Income Arbitrage` 在各自的“相关性簇”中方差较小，于是给予了合理的重仓（33% 和 20%）。同时，它给那些高波动、或高相关性的“拥挤赛道”（如 `Emerging Markets`）分配了用来控制风险底线的较低权重（1% 左右）。

## 3. 金融“族谱树” (Dendrogram) 可视化

HRP 之所以能如此聪明，是因为它在内部分三个步骤完成：
1. **层次聚类 (Tree Clustering)**：计算两两资产的距离，画出一棵树。
2. **拟对角化 (Quasi-Diagonalization)**：将树图压扁，让相关性最强的资产挨在一起。
3. **递归二分 (Recursive Bisection)**：从树根开始，每次分叉处，根据左右两个子分支内部的方差总和，按反比分配资金。

我们的底层包含了一个极其强大的可视化工具，让你一眼看清这 13 个策略的真实“血缘关系”：

```python
from pyfolioanalytics.plots import plot_dendrogram

# 绘制交互式的金融聚类树图
fig = plot_dendrogram(returns, linkage_method="single")
fig.show()
```
运行上述代码，浏览器会弹出一张形似基因图谱的结构图。你会惊讶地发现，机器完全不需要人工打标签，就能通过收益率的跳动，自动把 `Event Driven` 和 `Merger Arbitrage` （事件驱动与并购套利，在现实中确实是一套体系）聚成了最近的“亲兄弟”。

## 总结

当遇到维度爆炸、相关性陷阱、或是对传统凸优化的“梭哈现象”感到失望时，请将 `optimize_method` 切换为 `"HRP"` 乃至更先进的 `"HERC"` 或 `"NCO"`。

机器学习在量化配置中大放异彩的时代已经到来，而您手中掌握的，正是这套新时代利器的核心钥匙。
