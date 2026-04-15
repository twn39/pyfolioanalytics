# 进阶教程 5：对抗黑天鹅与回撤 (Advanced Risk Measures)

如果回顾前四篇教程，我们会发现所有的优化都建立在同一个风险指标上：**`StdDev` (标准差/波动率)**。

这也是经典马科维茨（Markowitz）均值-方差模型的基石。但方差模型有一个致命的缺陷：它假设**收益率是正态分布的，并且认为“向上涨”和“向下跌”同等危险**。

在现实的对冲基金中，我们并不害怕股票向上暴涨带来的“波动”，我们只害怕它**向下暴跌**，以及随之而来的**深度套牢（Drawdown）**。

本篇教程将教你如何使用 `PyFolioAnalytics` 强大的非对称凸优化引擎，将你的防守重心从“平庸的波动”转移到“致命的黑天鹅”。

## 1. 认识非对称风险

我们以两只完全不同的股票为例：
- A 股票：每天在 +1% 和 -1% 之间震荡。
- B 股票：每天涨 2%，但每年必定有一天暴跌 -20%（黑天鹅）。

在传统的“最小方差”模型眼里，这两只股票可能风险是一样的（甚至 B 的方差更大），但在真实的投资者眼里，B 股票拥有致命的**尾部风险（Tail Risk）**。

我们需要更高级的武器：
- **ES / CVaR (条件在险价值)**：只看历史上最糟糕的 5% 的日子，求那 5% 糟糕日子的平均亏损。
- **CDaR (条件最大回撤)**：只看历史上最惨烈的 5% 的回撤深度，求平均套牢幅度。

## 2. 声明不同的风险目标

在 `PyFolioAnalytics` 中，切换风险引擎只需要更改 `add_objective` 中的名字，底层会自动组装出对应的高维线性/二阶锥凸优化等式。

让我们读取过去 3 年（2023-2026）的 FAAMG 股票数据，保持满仓和只做多，跑三个不同目标的组合：

```python
import pandas as pd
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio
from pyfolioanalytics.risk import ES, CDaR

returns = pd.read_csv("data/stock_returns.csv", index_col="Date", parse_dates=True)
port = Portfolio(assets=list(returns.columns))
port.add_constraint(type="weight_sum", min_sum=1.0, max_sum=1.0)
port.add_constraint(type="long_only")

# ---------------------------------------------
# 方案 1：传统的最小方差 (StdDev)
# ---------------------------------------------
port_std = port.copy()
port_std.add_objective(type="risk", name="StdDev")
w_std = optimize_portfolio(returns, port_std, optimize_method="ROI")['weights']

# ---------------------------------------------
# 方案 2：抗黑天鹅的条件在险价值 (CVaR/ES)
# ---------------------------------------------
port_cvar = port.copy()
# arguments={"p": 0.95} 表示我们只防范最糟糕的 5% 尾部行情
port_cvar.add_objective(type="risk", name="ES", arguments={"p": 0.95})
w_cvar = optimize_portfolio(returns, port_cvar, optimize_method="ROI")['weights']

# ---------------------------------------------
# 方案 3：抗深度套牢的条件回撤 (CDaR)
# ---------------------------------------------
port_cdar = port.copy()
# 优化引擎将直接沿着时间轴计算水下回撤，并最小化前 5% 的深坑
port_cdar.add_objective(type="risk", name="CDaR", arguments={"p": 0.95})
w_cdar = optimize_portfolio(returns, port_cdar, optimize_method="ROI")['weights']
```

## 3. 结果解读：方差 vs 回撤

运行上述代码后，你会看到极其明显的仓位分化：

```text
股票       | [传统]方差       | [防崩]CVaR     | [防套牢]CDaR   
-------------------------------------------------------
AAPL     |     28.21% |     29.13% |     81.51%
AMZN     |      9.88% |      8.46% |     -0.00%
GOOGL    |     23.85% |     23.77% |      0.00%
META     |      0.01% |      0.00% |     -0.00%
MSFT     |     38.05% |     38.65% |     18.49%
-------------------------------------------------------
```

**市场解读**：
1. **方差 与 CVaR 的微调**：对于大盘科技股而言，它们的分布其实比较接近正态。但是在考虑了极端黑天鹅的 CVaR 视角下，AMZN 的权重进一步遭到了削弱，补充到了 AAPL 身上。
2. **CDaR 的极致避险**：这是全场最震撼的结果！当目标从“躲避单日下跌”变成“躲避长期套牢回撤”时，优化器惊人地将 **81.51%** 的资金全砸给了苹果 (AAPL)！
   - 因为回顾过去三年，虽然别家的日均波动也不算离谱，但在财报季的**连跌套牢周期（最大回撤持续时间）**中，苹果是唯一具有极强复苏反弹韧性的品种。如果你极度害怕买入后被套牢半年，CDaR 模型会毫不犹豫地让你抱紧苹果。

## 4. 事后归因与风险度量验证

这到底是不是玄学？我们可以调用 `pyfolioanalytics.risk` 中的离线验证函数，直接回测这三种配置在这三年里的**真实极限回撤**：

```python
# 事后风险验证
risk_cdar_std = CDaR(w_std.values, returns.values, p=0.95)
risk_cdar_cdar = CDaR(w_cdar.values, returns.values, p=0.95)

print(f"传统的方差组合，事后 95% CDaR: {risk_cdar_std:.2%}")
print(f"专防套牢的 CDaR 组合，事后 95% CDaR: {risk_cdar_cdar:.2%}")
```

打印出的验证结果：
```text
传统的方差组合，事后 95% CDaR: 30.85%
专防套牢的 CDaR 组合，事后 95% CDaR: 25.11%
```

如果你使用了传统的方差组合，在你运气最差的 5% 的时间里，你的账户会面临高达 **-30.85%** 的恐怖回撤（这足以让大部分投资人崩溃清仓）。
但因为我们在模型 3 中直接启用了 `CDaR` 最优化，引擎通过极端的仓位调整，硬生生把这历史级别的深坑填高了 5 个点，回撤被强力控制在了 **-25.11%**。

## 总结

你刚刚见识到了量化界“高阶水流体动力学”的威力。

在 `PyFolioAnalytics` 中，摆脱传统的平庸方差只需要改一个字符串：
- `name="ES"` / `name="CVaR"`: 害怕单日黑天鹅？切到这个！
- `name="MAD"` / `name="semi_MAD"`: 想要忽略偶尔的毛刺波动？切到这个！
- `name="CDaR"`: 极度害怕被高位套牢、想要最平滑的水下净值曲线？切到这个！

本系列的新手进阶教程至此告一段落。相信你已经掌握了如何用这个强大的底层库，组装出符合各种严苛条件的工业级投研策略。
