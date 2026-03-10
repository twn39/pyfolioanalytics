# PyFolioAnalytics

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Validation](https://img.shields.io/badge/Validation-R%20Parity%20Check-orange.svg)](tests/)

**PyFolioAnalytics** 是 R 语言顶级组合优化库 [PortfolioAnalytics](https://github.com/braverock/PortfolioAnalytics) 的高保真 Python 实现。本项目旨在为量化研究员提供一个模块化、可扩展且数学精确的框架，用于现代资产配置、风险归因和回测验证。

---

## 📖 核心功能深度解析

### 1. 灵活的投资组合规格 (Portfolio Specification)
我们复刻了 R 语言中“声明式”的组合定义方式，支持多种复杂的实务约束。

*   **基础约束**: `box` (上下限), `group` (行业/板块分组), `full_investment` (权重和为1), `dollar_neutral` (多空对冲)。
*   **因子暴露约束 (Factor Exposure)**: 
    *   直接限制组合在指定因子（如 Market, Value, Growth）上的 Beta 加权暴露。
    *   公式: $lower \le B^T w \le upper$。
*   **杠杆暴露约束 (Leverage Exposure)**: 
    *   控制组合的“毛杠杆”（Gross Leverage），支持 130/30 等策略。
    *   公式: $\sum |w_i| \le leverage\_limit$。
*   **多样化约束 (HHI)**: 
    *   利用赫芬达尔-赫希曼指数（HHI）防止权重过度集中。
    *   公式: $1 - \sum w_i^2 \ge div\_target$。
*   **交易成本与换手率**: 支持设置单边交易成本 `ptc` 和回测中的 `turnover_target`。

### 2. 高级矩估计 (Statistical Moment Estimation)
矩估计是组合优化的灵魂。本项目提供了多种先进技术来降低估计误差（Estimation Error）。

*   **CCC-GARCH 动态波动率**:
    *   利用单变量 GARCH(1,1) 模型预测时变的条件波动率，配合常相关性矩阵（CCC）构造下一期的协方差矩阵。
    *   适用于波动率聚集（Volatility Clustering）明显的金融市场。
*   **Meucci 熵池法 (Entropy Pooling)**:
    *   允许以非参数方式将主观观点（View）整合进先验分布。
    *   支持解析梯度加速优化，能够处理包含数千个场景的复杂观点。
*   **鲁棒协方差 (Robust Covariance)**:
    *   集成 `MinCovDet` (MCD) 和 `Ledoit-Wolf` 收缩估计，处理金融数据中的离群值。
*   **随机矩阵理论 (RMT) 去噪**:
    *   通过 Marchenko-Pastur 分布过滤协方差矩阵中的噪声特征值，提取真实的结构性相关性。

### 3. 多样化的优化引擎
*   **凸优化 (CVXPY)**: 核心引擎，支持均值-方差（MVO）、二阶锥规划（SOCP）和混合整数规划（MILP）。
*   **风险平价 (Risk Parity)**: 
    *   **ERC**: 支持非线性求解器精确实现等风险贡献。
    *   **HRP/HERC**: 基于分层聚类（Hierarchical Clustering）的稳健分配，无需协方差矩阵求逆。
*   **Critical Line Algorithm (CLA)**: 专门用于精确描绘包含禁止做空约束的均值-方差有效前沿。

### 4. 风险度量与归因 (Risk Attribution)
*   **尾部风险**: 除了波动率，还支持 CVaR (ES)、Modified VaR、CDaR (最大回撤风险) 和 EVaR (熵风险)。
*   **风险分解**: 
    *   **资产级**: 提供边际风险贡献 (MCR)、成分风险贡献 (CCR) 和百分比贡献 (PCR)。
    *   **因子级**: 将总风险归因为系统性因子贡献和特质性残余贡献。

---

## 🔬 数学保真度与交叉验证

我们坚持“结果必须与 R 完全一致”的原则。

*   **基准数据集**: 所有的算法均在 `edhec`（对冲基金策略指数）和真实 A 股/美股数据上进行了验证。
*   **自动化测试**: `tests/` 目录包含超过 100 个交叉验证用例。我们通过运行 R 脚本生成 JSON 基准数据，并在 Python 中进行 $10^{-7}$ 精度的断言。
*   **透明性**: 项目内置 `scripts/` 目录，您可以随时运行 `generate_*.R` 脚本来复现 R 语言的原始计算过程。

---

## 🛠 快速上手

### 安装
推荐使用现代 Python 包管理器 `uv`：
```bash
uv sync
```

### 示例：带因子约束的 GARCH 优化
```python
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio
import numpy as np
import pandas as pd

# 1. 准备数据
R = pd.read_csv("data/edhec.csv", index_col=0)
B = np.random.rand(R.shape[1], 2) # 假设的因子加载矩阵

# 2. 构建组合规格
p = Portfolio(assets=R.columns)
p.add_constraint(type="full_investment")
p.add_constraint(type="factor_exposure", B=B, lower=[0.2, 0.1], upper=[0.5, 0.4])
p.add_constraint(type="HHI", hhi_target=0.15)

# 3. 执行优化 (使用 GARCH 矩估计)
res = optimize_portfolio(R, p, moment_method="garch")

# 4. 风险分解
from pyfolioanalytics.risk import risk_decomposition
decomp = risk_decomposition(res['weights'], res['moments']['sigma'])
print(f"资产百分比风险贡献: {decomp['pcr']}")
```

---

## 📂 项目结构
```text
pyfolioanalytics/
├── data/           # 交叉验证基准 (JSON) 与真实数据集 (CSV)
├── scripts/        # 地面真理 (Ground Truth) 生成脚本 (R/Python/Julia)
├── src/            # 核心库源码
│   ├── moments.py  # 矩估计 (GARCH, 收缩, RMT)
│   ├── portfolio.py# 组合定义与约束系统
│   ├── risk.py     # 风险度量与归因
│   └── solvers.py  # 优化求解器集成
├── tests/          # 完整的回归测试套件
└── third_party/    # 参考库源码 (不提交，仅供开发对照)
```

---
*量化投资，始于精确。*
