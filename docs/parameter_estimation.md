# 参数与矩估计 (`moments.py` & `rmt.py`)

在进行现代投资组合优化之前，必须对资产收益率分布的“矩”（Moments，如均值、协方差矩阵、高阶矩）进行估计。本模块提供了从经典到前沿的多种估计手段。

## 核心 API

### `set_portfolio_moments(R, portfolio, method="sample", **kwargs)`
根据指定的统计学方法从历史收益率数据中估计参数。

- **输入**:
  - `R` (pd.DataFrame): $T \times N$ 的历史收益率矩阵。
  - `portfolio` (Portfolio): 组合规格实例。
  - `method` (str): 均值/协方差估计方法。可选值包括：
    - `"sample"`: 传统样本协方差与算术平均。
    - `"robust"`: 最小协方差决定 (MCD) 稳健估计。
    - `"factor_model"`: 统计因子模型 (PCA 降维重建)。
    - `"denoised"`: Marchenko-Pastur 随机矩阵去噪 (RMT)。
    - `"ewma"`: 指数加权移动平均 (对近期数据赋予更高权重)。
    - `"semi_covariance"`: 下行半协方差矩阵 (惩罚跌破基准的波动)。
    - `"garch"`: CCC-GARCH(1,1) 动态条件协方差。
  - `comoment_method` (str, 可选): 高阶矩 (共偏度、共峰度) 估计法，如 `"shrinkage"` (Ledoit-Wolf 收缩)。
  - `**kwargs`: 对应方法的超参数 (如 `span=36` 给 ewma，`benchmark=0.0` 给 semi_covariance)。

- **输出**:
  - `Dict[str, Any]`: 包含估计结果的字典，通常具有 `"mu"`, `"sigma"`, `"m3"`, `"m4"` 键。

### RMT 去噪辅助函数 (`rmt.py`)
- `denoise_covariance(sigma, q, method="fixed")`: 基于随机矩阵理论对相关性/协方差矩阵进行去噪（截断或收缩噪音特征值）。
- `gerber_statistic(R, threshold=0.5)`: 计算 Gerber 稳健共变统计量，用于极端市况下的协方差替代。

## 代码示例

```python
import pandas as pd
import numpy as np
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.moments import set_portfolio_moments

# 构造模拟收益率数据
np.random.seed(42)
R = pd.DataFrame(np.random.randn(100, 3) * 0.01 + 0.001, columns=['Asset1', 'Asset2', 'Asset3'])
port = Portfolio(assets=list(R.columns))

# 方法 1：传统的样本均值与协方差
moments_sample = set_portfolio_moments(R, port, method="sample")
print("Sample Covariance:\n", moments_sample["sigma"])

# 方法 2：使用 EWMA (指数加权) 估计近期波动，衰减跨度 36 期
moments_ewma = set_portfolio_moments(R, port, method="ewma", span=36)

# 方法 3：使用随机矩阵理论 (RMT) 去噪，常用于处理 N > T 或高噪数据
moments_rmt = set_portfolio_moments(R, port, method="denoised", denoise_method="spectral")

# 方法 4：计算包含高阶矩的收缩估计（用于 Modified VaR 优化）
moments_higher = set_portfolio_moments(
    R, port, 
    method="sample", 
    comoment_method="shrinkage", 
    comoment_alpha=0.5
)
print("Coskewness Matrix Shape:", moments_higher["m3"].shape) # (N, N^2)
```