# 构建与验证量化投资组合的标准流程

在 `PyFolioAnalytics` 中，我们提倡一种严谨、可验证的“声明式”组合构建范式。从原始数据到实盘部署，一个完整的量化投资组合配置流程通常分为四个标准阶段：**数据处理与参数估计**、**模型声明**、**优化求解**、**回测与量化归因**。

## 阶段 1: 收益率提取与前瞻性参数估计 (Parameter Estimation)

任何优化器在底层都是对输入的**预期收益率向量 ($\mu$)** 和**协方差矩阵 ($\Sigma$)** 进行数学规划。如果这些输入是带噪的，优化出的权重将毫无意义（即“垃圾进，垃圾出”）。

### 标准步骤：
1. **准备数据**: 收集一篮子资产的日度或月度历史收益率序列（通常为 `pd.DataFrame` 格式，行是时间，列是资产名）。
2. **选择去噪/降维方法**:
   - 如果你的资产池庞大（例如标普 500）但历史窗口短，直接计算协方差矩阵不可逆。应该使用 `method="denoised"` (RMT 去噪) 或 `method="factor_model"`（统计主成分降维）。
   - 如果你想让近期发生的事件（如突发黑天鹅）在矩阵里占比更高，使用 `method="ewma"`（指数加权移动平均）。
   - 如果你想融入宏观研究员的主观观点（比如“我认为科技股下个月会跑赢金融股 2%”），则使用 Black-Litterman 或 Meucci 熵池法计算后验矩阵。

```python
from pyfolioanalytics.moments import set_portfolio_moments

# 以 RMT (随机矩阵理论) 去除相关性矩阵里的纯噪音特征值
moments = set_portfolio_moments(R, port, method="denoised", denoise_method="fixed")
```

---

## 阶段 2: 声明式组合约束与目标 (Portfolio Specification)

不要在求解器里手写枯燥的线性代数公式！你需要定义这个资金池的客观业务边界，并告诉算法你的核心诉求。

### 标准步骤：
1. **划定边界 (Constraints)**: 
   - 资金能打满吗？(`full_investment`)
   - 允许做空吗？(`long_only` 或 `dollar_neutral`)
   - 风险合规部门有硬性持仓限制吗？(`box` 控制单股最高 15%，`group` 控制半导体板块不超过 40%)。
2. **设定目标 (Objectives)**: 
   - 你追求绝对的高收益还是下行保护？
   - 结合上一步处理好的分布特性，你可以声明 `name="StdDev"` 最小化波动率，或者更激进地声明 `name="CVaR"` 惩罚极端尾部风险，又或者声明 `name="MAD"` 做稳健的绝对偏差最小化。

```python
from pyfolioanalytics.portfolio import Portfolio

port = Portfolio(assets=list(R.columns))
port.add_constraint(type="full_investment")
port.add_constraint(type="long_only")
port.add_constraint(type="box", min=0.01, max=0.15) # 单只股票占比 1%~15%

# 目标：在满足上述条件的情况下，追求给定预期收益下的 CVaR 风险最小化
port.add_objective(type="risk", name="CVaR", arguments={"p": 0.95})
```

---

## 阶段 3: 寻找全局最优解 (Optimization)

一旦业务逻辑被声明，下一步是调用数学规划引擎。

### 标准步骤：
1. 调用核心引擎 `optimize_portfolio`。
2. 引擎会自动分析你声明的目标和约束，将它们编译为特定的凸优化问题（例如：二次规划 QP 对应方差，线性规划 LP 对应 MAD/CVaR，指数锥规划对应熵风险 EVaR 等），并喂给 `CVXPY` 求解器。
3. 检查返回的 `status`。如果发生“不可能三角”（比如你要求每只股票最低占比 20%，但总和又要求等于 100%，而你有 10 只股票），求解器会准确抛出 `infeasible` 状态。

```python
from pyfolioanalytics.optimize import optimize_portfolio

res = optimize_portfolio(R, port, optimize_method="ROI", moments=moments)

if res["status"] == "optimal":
    print("找到最优解！权重分布为：")
    print(res["weights"])
```

---

## 阶段 4: 滚动回测与量化归因 (Validation & Attribution)

即使某一天找到了完美的最优解，在真实市场中随着价格波动，这套体系是否经得起时间的考验？你需要将整个步骤 1-3 放入时间切片机中验证。

### 标准步骤：
1. **Walk-forward 回测**: 定义一个滚动窗口（例如：过去 250 天数据作为训练集估计参数并优化出当月权重，持仓 1 个月后再次重估，不断向前推进）。这由 `optimize_portfolio_rebalancing` 自动管理。
2. **生成交互式报表**: 将回测产生的真实历史净值与调仓权重传递给可视化引擎。
   - 用 `plot_performance` 验证总收益与最大回撤（Underwater Drawdown）。
   - 用 `plot_weights` 查看历史仓位变动是否剧烈（如果太过频繁，说明需要回滚到阶段 2，加入 `turnover` 换手率约束）。
   - 用 `plot_risk_decomposition` 计算事后风控：是否看似分散的资金，实际 90% 的风险都来自于英伟达单只股票（风险过载）？

```python
from pyfolioanalytics.backtest import optimize_portfolio_rebalancing
from pyfolioanalytics.plots import plot_performance, plot_weights, plot_risk_decomposition

# 执行滚动再平衡（每月调仓，使用过去1年的数据计算协方差）
bt_result = optimize_portfolio_rebalancing(
    R_history, port, 
    optimize_method="ROI",
    training_period=252, 
    rebalance_on="months"
)

# 提取回测表现与权重流
returns = bt_result.extract_returns()
weights_df = bt_result.extract_weights()

# 图表归因
fig_perf = plot_performance(returns, title="策略历史净值与回撤")
fig_perf.show()
```

### 总结
这就是在 `PyFolioAnalytics` 中从零构建到上线部署的无缝流程：**抗噪参数提取 -> 声明业务规则 -> 凸优化求解 -> 滚动时间机器验证与归因。**
