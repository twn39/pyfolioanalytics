# 风险度量 (`risk.py`)

本模块包含所有受支持的量化风险测度的底层数学实现。所有函数均设计为矢量化操作，能够接收投资组合的权重向量并返回相应的风险标量值。

## 核心 API 与风险函数

所有主要函数的签名基本一致：
- `measure_name(weights: np.ndarray, R: np.ndarray, **kwargs) -> float`
或对于闭式参数模型：
- `measure_name(weights: np.ndarray, mu: np.ndarray, sigma: np.ndarray, ...) -> float`

### 离散与鲁棒估计 (Robust Measures)
- **`MAD(weights, R)`**: 平均绝对偏差 (Mean Absolute Deviation)。计算收益率偏离均值的绝对距离的平均值。对异常值极不敏感。
- **`semi_MAD(weights, R)`**: 下行平均绝对偏差。仅惩罚低于均值的收益率部分。
- **`l_moment(R, weights, k=2)`**: 计算组合收益率分布的第 $k$ 阶 L-矩。

### 尾部风险与分位数风险 (Quantile Risks)
- **`VaR(weights, mu, sigma, m3, m4, p=0.95, method="gaussian")`**: 风险价值。支持 `"gaussian"`, `"historical"`, 及基于 Cornish-Fisher 展开的 `"modified"` 肥尾调整法。
- **`ES(weights, mu, sigma, m3, m4, p=0.95, method="gaussian")`**: 预期短缺 (CVaR/条件 VaR)。计算尾部超过 VaR 部分的平均损失。
- **`EVaR(weights, R, p=0.95)`**: 熵风险价值 (Entropic Value at Risk)。VaR 的最紧凸上限，基于 Chernoff 不等式。
- **`RLVaR(weights, R, p=0.95, kappa=0.3)`**: 相对论风险价值 (Relativistic VaR)。

### 回撤度量 (Drawdown Risks)
- **`calculate_drawdowns(p_returns)`**: 返回时间序列数组的各期回撤百分比。
- **`max_drawdown(weights, R)`**: 绝对最大回撤水位。
- **`average_drawdown(weights, R)`**: 历史序列平均回撤。
- **`CDaR(weights, R, p=0.95)`**: 条件回撤风险。最差的 (1-p) 比例回撤幅度的平均值。
- **`EDaR(weights, R, p=0.95)`**: 熵回撤风险。
- **`RLDaR(weights, R, p=0.95, kappa=0.3)`**: 相对论回撤风险。

### 风险分解 (Risk Attribution)
- **`risk_contribution(weights, sigma)`**: 计算各个资产对投资组合总体方差的边际风险贡献 (Marginal Risk Contribution, MRC)。用于实现风险平价配置。

## 代码示例

```python
import numpy as np
import pandas as pd
from pyfolioanalytics.risk import MAD, semi_MAD, ES, CDaR, risk_contribution

# 随机模拟收益
np.random.seed(42)
R = np.random.randn(100, 3) * 0.02
weights = np.array([0.4, 0.4, 0.2])

# 测度计算
print("MAD:", MAD(weights, R))
print("Semi-MAD:", semi_MAD(weights, R))
print("Historical CVaR (95%):", ES(weights, None, None, p=0.95, method="historical", R=R))
print("CDaR (95% worst drawdowns):", CDaR(weights, R, p=0.95))

# 计算方差边际风险贡献
sigma = np.cov(R, rowvar=False)
mrc = risk_contribution(weights, sigma)
print("Marginal Risk Contributions:", mrc)
```