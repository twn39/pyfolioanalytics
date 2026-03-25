# 机器学习与图论配置 (`ml.py` & `dbht.py`)

传统均值-方差优化 (MVO) 对协方差矩阵求逆时高度敏感，极易放大估计误差。本模块提供了一系列基于**图论拓扑聚类**和**层次聚类**的无偏机器学习资金配置模型，能在不直接求逆的情况下解决强相关性资产带来的权重极端化问题。

## 核心 API

所有函数的输入均为资产的协方差矩阵或收益率相关数据，输出为标准化资产权重（`pd.Series`）。

### `hrp_portfolio(sigma, linkage_method='ward')`
**分层风险平价 (Hierarchical Risk Parity, HRP)**
1. 基于协方差/相关性矩阵计算资产间距离。
2. 使用层次聚类将资产归类并构建树状图。
3. 执行矩阵的拟对角化。
4. 递归的等分二分法自上而下分配风险。

### `herc_portfolio(sigma, linkage_method='ward', risk_measure='variance', R=None)`
**分层等风险贡献 (Hierarchical Equal Risk Contribution, HERC)**
相比 HRP，HERC：
- 在聚类分配树节点时，不仅仅使用方差（`variance`），还支持 `std_dev`, `mad`, `cvar`, `cdar` 等尾部和绝对偏差风险测度（需同时传入 `R`）。
- 允许根据聚类树剪枝后，在同簇内部进一步执行风险平价或等权配置。

### `nco_portfolio(mu, sigma, max_clusters=None)`
**嵌套聚类优化 (Nested Clustered Optimization, NCO)**
1. 利用 KMeans 或轮廓系数聚类将矩阵分解为弱相关的子块。
2. 在每个强相关的子块（Cluster）内部运行传统的局域最优配置。
3. 将每个簇视为单一宏观资产，在簇间执行全局风险平价组合，彻底剥离协方差矩阵病态特征值的影响。

### `dbht.py` 模块
实现 **有向加权图与最大平面子图 (Directed Bubble Hierarchical Tree, DBHT)**：
提取复杂的拓扑数据网络过滤方法，可以用 `method='dbht'` 替代传统 `scipy` 的层次聚类。

## 代码示例

```python
import pandas as pd
import numpy as np
from pyfolioanalytics.ml import hrp_portfolio, nco_portfolio

# 构造具有特定相关性组块的数据
np.random.seed(42)
T, N = 200, 6
factor1 = np.random.randn(T, 1)
factor2 = np.random.randn(T, 1)

R_data = np.hstack([
    factor1 + np.random.randn(T, 3)*0.5, # 前3个资产高相关
    factor2 + np.random.randn(T, 3)*0.5  # 后3个资产高相关
]) * 0.01

R = pd.DataFrame(R_data, columns=[f"A{i}" for i in range(1, 7)])
sigma = R.cov()
mu = R.mean().values

# 1. 运行 HRP 
w_hrp = hrp_portfolio(sigma.values, linkage_method='ward')
print("HRP Weights:\n", pd.Series(w_hrp, index=R.columns))

# 2. 运行 NCO (嵌套聚类优化)
w_nco = nco_portfolio(mu, sigma.values, max_clusters=2)
print("NCO Weights:\n", pd.Series(w_nco, index=R.columns))
```