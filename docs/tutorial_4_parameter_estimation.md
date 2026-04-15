# 进阶教程 4：参数降噪与处理“垃圾数据” (Parameter Estimation)

在量化投资界有一句著名的话：**"Garbage In, Garbage Out"** (垃圾进，垃圾出)。

在前三篇教程中，我们都在教你如何构建复杂的规则（目标、约束、滚动回测），但这些规则都有一个共同的前提：它们**极度依赖**底层的“协方差矩阵”（也就是每两只股票的历史相关性与波动率）。

如果你直接将过去的每日收益率取个简单的平均和方差喂给机器，一旦历史中出现一次因为“财报暴雷”导致的 -20% 极端跌幅，优化器就会被“吓坏”，在未来的好几年里都将这只股票拉黑。又或者，最近两个月行情已经发生了巨大的风格切换，但由于 3 年的数据均值效应，优化器依然无动于衷。

本篇教程将展示 `PyFolioAnalytics` 在数据处理（Parameter Estimation）上强大的“净水器”能力。

## 1. 对比基准：原始的样本协方差

我们再次以过去 3 年（2023-2026）的 FAAMG 科技股作为数据，构建一个无任何高级约束的 **“最小方差组合”**。

如果使用最基础的数学定义计算协方差（也就是假设过去 750 天每一天都同等重要，且没有任何数据清洗），代码与结果如下：

```python
import pandas as pd
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

returns = pd.read_csv("data/stock_returns.csv", index_col="Date", parse_dates=True)
port = Portfolio(assets=list(returns.columns))
port.add_constraint(type="weight_sum", min_sum=1.0, max_sum=1.0)
port.add_constraint(type="long_only")
port.add_objective(type="risk", name="StdDev")

# [基准] 传统样本协方差 (Sample Covariance)
res_sample = optimize_portfolio(returns, port, moment_method="sample")

# 打印结果：
# AAPL: 28.21% | AMZN: 9.88% | GOOGL: 23.85% | META: 0.00% | MSFT: 38.05%
```
在基准结果中，META（由于前几年的一些暴跌历史导致方差过大）被直接分配了 0 权重，而 MSFT 因为长期稳健拿到了最多的 38.05%。

下面我们看看如果不改模型，仅仅改变**数据降噪/提取手法**，世界会发生怎样的变化。

## 2. 拥抱近期记忆：指数加权 (EWMA)

在实际炒股时，“上个月的波动”对下个月的指导意义，显然远远大于“3 年前某个月的波动”。

我们可以使用 `moment_method="ewma"`（指数加权移动平均）。通过设置 `span=60`（大约 3 个月），我们让最近的 60 个交易日在矩阵中占据绝对的权重主导：

```python
# [净化手法 1] EWMA 指数加权（追踪当下行情）
res_ewma = optimize_portfolio(
    returns, port, 
    moment_method="ewma", 
    span=60
)

# 打印结果：
# AAPL: 29.19% | AMZN: 0.00% | GOOGL: 34.77% | META: 11.74% | MSFT: 24.30%
```

**市场解读**：神奇的现象出现了！META 虽然在 3 年的长周期里波动极大（导致基准给了 0），但在**最近的几个月（span=60）**里，它其实非常平稳甚至走出了连涨趋势。于是，EWMA 敏锐地抓住了这个“近期特征”，将 META 的配置直接上调到了 **11.74%**！而之前最稳妥的 MSFT 可能近期波动加剧，反而被砍到了 24.30%。

## 3. 剔除黑天鹅：Boudt 稳健清洗 (Robust Cleaning)

有时候，股票突然出现一个 -15% 的单日跌幅，可能只是因为 CEO 在社交媒体上发了一条错误的帖子（或者财报的偶发性错杀），第二天就涨回去了。但在统计学上，这种**肥尾极端值（Outlier）**会把方差拉爆。

`PyFolioAnalytics` 内置了学术界著名的 `Boudt` 清洗算法。只要加上参数，它就会像高级过滤网一样，**自动侦测并“削平”这种极端的长尾数据，还原股票正常的波动特征。**

```python
# [净化手法 2] Boudt 清洗（自动剔除财报暴雷、闪崩等黑天鹅异常点）
res_boudt = optimize_portfolio(
    returns, port, 
    moment_method="sample", 
    clean_returns="boudt", 
    clean_alpha=0.05 # 定义 5% 的极值区域进行削平处理
)

# 打印结果：
# AAPL: 29.41% | AMZN: 1.91% | GOOGL: 22.86% | META: 0.00% | MSFT: 45.81%
```

**市场解读**：我们发现，AMZN（亚马逊）的配置比例从基准的 9.88% 骤降到了 1.91%。这说明在过滤掉单日的黑天鹅之后，系统认为 AMZN 的“正常波动率”依然偏高；而微软 (MSFT) 的正常交易日极其稳定，在去除了极少数暴跌日的干扰后，它的可靠性被进一步确认，权重史无前例地加到了 **45.81%**。

## 4. 消除随机噪音：Ledoit-Wolf 压缩 (Shrinkage)

在机构量化中，如果你计算了几百只股票的协方差矩阵，里面一定充满了因为巧合导致的“伪相关性”（比如，由于随机波动，某只医药股和某只科技股在某几个月碰巧涨跌一致）。

如果放任不管，优化器（它是个极度聪明的数学偏执狂）一定会敏锐地抓住这个漏洞，把巨大的仓位重注在这个“伪相关”上，一旦实盘就会原形毕露。这就需要用到 **Ledoit-Wolf 压缩 (Shrinkage)** 算法。

```python
# [净化手法 3] 压缩矩阵（向常数相关性靠拢，防止过拟合）
res_shrink = optimize_portfolio(
    returns, port, 
    moment_method="shrinkage", 
    shrinkage_target="constant_correlation"
)

# 打印结果：
# AAPL: 27.66% | AMZN: 6.11% | GOOGL: 24.01% | META: 0.00% | MSFT: 42.22%
```

**市场解读**：你可以把 Shrinkage 视为机器学习里的 `L2 正则化`。它通过强行将协方差矩阵向一个“平庸的均值”靠拢，抹平了那些过于锋利的“棱角”。这看起来配置没有太大的戏剧性变化，但它在海量选股（比如标普 500）的实盘回测中，由于极大地消除了**过拟合**，其实盘胜率远高于裸样本协方差！

## 总结

优化器是一把锋利的宝剑，但如果喂给它带毒的数据，它一样会反噬你。

在 `PyFolioAnalytics` 中，你拥有量化界最齐全的数据净水组件：
- 要追随当前热点？用 `moment_method="ewma"`。
- 数据极度易受消息面冲击？用 `clean_returns="boudt"`。
- 选股池高达上千只导致维度诅咒？用 `moment_method="shrinkage"` 或 `"denoised"` (RMT 降噪)。

你可以在工作区中运行 `uv run python test_tutorial_4.py`，在一个脚本里一次性跑出这四种数据处理手法的对比。在下一节，我们将告别方差，进入非对称风险（最大回撤、条件在险价值）的领域！
