# PyFolioAnalytics

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Validation](https://img.shields.io/badge/Validation-R%20Parity%20Check-orange.svg)](tests/)

**PyFolioAnalytics** 是 R 语言顶级组合优化库 [PortfolioAnalytics](https://github.com/braverock/PortfolioAnalytics) 的高保真 Python 实现。本项目不仅完美复刻了 R 语言的声明式定义和优化逻辑，还整合了 `Riskfolio-Lib` 和 `PyPortfolioOpt` 中的现代量化特性，通过严苛的 $10^{-7}$ 精度交叉验证，为量化研究提供生产级的数学保障。

---

## 🛠 全功能概览

### 1. 组合规格与复杂约束 (Portfolio Specification)
复刻了 R 语言 `portfolio.spec` 的核心逻辑，支持高度灵活的约束组合：
-   **资产分配约束**: `box` (权重上下限), `group` (组约束), `position_limit` (最大持仓数)。
-   **预算与杠杆**: `full_investment` (权重和为1), `dollar_neutral` (多空对冲), `leverage_exposure` (毛杠杆限制，支持 130/30)。
-   **因子与风险**: 
    -   `factor_exposure`: 限制风格/宏观因子的 Beta 暴露。
    -   `tracking_error`: 限制相对于基准的跟踪误差。
    -   `active_share`: 强制组合与基准的主动管理差异。
    -   `diversification (HHI)`: 通过权重平方和限制实现多样化。
-   **交易控制**: `turnover` (换手率限制), `transaction_cost` (基于百分比的交易成本)。

### 2. 统计矩估计 (Statistical Moments)
提供多样化的均值、协方差及高阶矩估计技术，用于降低估计误差：
-   **动态波动率**: `CCC-GARCH` (单变量 GARCH + 常相关) 矩估计。
-   **主观观点整合**: `Meucci Entropy Pooling` (支持解析梯度) 与 `Black-Litterman`。
-   **稳健估计**: `MinCovDet` (鲁棒协方差), `Ledoit-Wolf` (收缩估计)。
-   **随机矩阵理论 (RMT)**: 特征值去噪（Fixed, Spectral, Shrink 方法）。
-   **高阶矩**: 样本及统计因子模型（SFM）生成的 **共偏度 (M3)** 和 **共峰度 (M4)** 矩阵。

### 3. 风险度量与归因 (Risk Analysis)
涵盖从经典波动率到前沿稳健度量的全方位分析：
-   **经典与修正风险**: Gaussian/Modified (Cornish-Fisher) VaR 和 ES (CVaR)。
-   **回撤风险**: `CDaR` (条件回撤), `EDaR` (熵回撤), `MaxDrawdown`。
-   **现代稳健度量**: `EVaR` (熵 VaR), `RLVaR/RLDaR` (基于稳健线性规划的 VaR/DaR)。
-   **排序加权 (OWA)**: 支持 GMD、L-Moments 权重、CRM 权重等 OWA 风险度量。
-   **风险归因**: 资产级与因子级的 **MCR** (边际), **CCR** (成分), **PCR** (百分比) 贡献分解。

### 4. 优化算法与引擎 (Optimization)
集成多种求解器，适配线性、二次及非凸优化：
-   **凸优化集成**: 基于 **CVXPY** 的 MVO, SOCP, MILP 求解。
-   **风险平价**: `ERC` (等风险贡献) 的非线性精确求解。
-   **有效前沿**: **CLA** (关键线算法) 实现带约束的精确前沿描绘。
-   **机器学习优化**: 
    -   `HRP` (分层风险平价), `HERC` (分层等风险贡献), `NCO` (嵌套聚类优化)。
    -   支持 **DBHT** (有向气泡层次树) 聚类算法。
-   **离散化分配**: 支持将百分比权重转换为实际股票股数的离散分配算法。

### 5. 架构与实务功能
-   **回测引擎**: 支持滚动窗口（Rolling）和扩张窗口（Expanding）的自动化调仓测试。
-   **层次化结构**: 支持 **Regime Switching** (状态切换) 组合和多层级嵌套组合。
-   **随机组合**: 支持基于 Simplex 变换的随机组合生成，用于非凸空间探索。

---

## 🔬 数学保真度与交叉验证

本项目每一项核心算法均通过了与 R 语言 `PortfolioAnalytics` 原生库的对比验证：
-   **验证数据集**: 包含 `edhec` 策略数据及 2020-2026 年真实股票数据。
-   **自动化断言**: 在 `tests/` 下有超过 100 个测试用例，确保 Python 与 R 的计算结果在 `1e-7` 精度下一致。
-   **透明逻辑**: 所有基准生成的 R 脚本均保留在 `scripts/` 目录，供用户复现。

---

## 📂 项目结构
-   `src/pyfolioanalytics/`:
    -   `moments.py`: 所有矩估计逻辑（GARCH, RMT, Shrinkage）。
    -   `risk.py`: 风险度量（VaR, ES, OWA, RLVaR）与归因分解。
    -   `meucci.py`: 熵池法与排名算法。
    -   `ml.py` & `dbht.py`: HRP, HERC, NCO 与 DBHT 聚类。
    -   `portfolio.py`: 组合规格定义与约束管理。
    -   `backtest.py`: 调仓回测引擎。
    -   `solvers.py`: CVXPY, SciPy, CLA 求解器接口。
-   `scripts/`: 地面真理 (Ground Truth) 生成脚本。
-   `data/`: 交叉验证 JSON 基准与数据集。

---
*量化研究的精准桥梁。*
