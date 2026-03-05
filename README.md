# PyFolioAnalytics

Python implementation of the R package [PortfolioAnalytics](https://github.com/braverock/PortfolioAnalytics).

## Features

- [x] **Portfolio Specification**: Support for Box, Group, Turnover, Transaction Costs, and Position Limit constraints.
- [x] **Optimization Engines**:
    - **CVXPY**: Linear, Quadratic (MVO), and Mixed-Integer programming.
    - **SciPy (SLSQP)**: Non-linear optimization for Equal Risk Contribution (ERC).
    - **Differential Evolution**: Global heuristic search for non-convex problems.
- [x] **Risk Modeling**:
    - Gaussian and Modified (Cornish-Fisher) VaR and ES.
    - Path-dependent measures: MaxDrawdown and AverageDrawdown.
- [x] **Statistical Models**:
    - Black-Litterman posterior estimation.
    - Statistical Factor Models (PCA).
    - Meucci Entropy Pooling for view integration.
- [x] **Backtesting**: Rolling-window and expanding-window rebalancing with flexible frequencies.
- [x] **Hierarchical Structures**: Support for Regime Switching and Multi-layer portfolio architectures.

## Installation

```bash
uv sync
```

## Testing & Validation

This library has been rigorously cross-validated against the original R `PortfolioAnalytics` and `PerformanceAnalytics` libraries using:
1.  **EDHEC Dataset**: Benchmark hedge fund index data.
2.  **Real Stock Data**: AAPL, MSFT, GOOGL, AMZN, META (2020-2026).
3.  **Macro Asset Data**: SPY, QQQ, GLD, TLT, BRK.B (2020-2026).

To run the parity tests:
```bash
uv run pytest
```

## Structure

- `src/pyfolioanalytics/`: Core package source.
- `data/`: Sample datasets (EDHEC, Real Stock returns).
- `tests/`: Comprehensive test suite including multi-dataset cross-validation.
- `third_party/PortfolioAnalytics/`: Original R source for reference.
