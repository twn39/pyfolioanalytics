library(PortfolioAnalytics)
library(jsonlite)
library(corpcor)
library(xts)

set.seed(42)
n_assets <- 5
n_obs <- 50
# Create random returns with dates for xts
dates <- seq(as.Date("2020-01-01"), length.out=n_obs, by="day")
R_mat <- matrix(rnorm(n_obs * n_assets), nrow=n_obs, ncol=n_assets)
colnames(R_mat) <- paste0("A", 1:n_assets)
R <- xts(R_mat, order.by=dates)

results <- list()

# 1. Ledoit-Wolf Shrinkage (corpcor::cov.shrink)
# Standard constant correlation target is common in R's corpcor
lw_res <- corpcor::cov.shrink(R)
results$shrinkage_sigma <- as.matrix(lw_res[1:n_assets, 1:n_assets])
results$returns <- as.matrix(R)

# 2. Active Ranking (ac.ranking)
# Ranking: A2 < A3 < A1 < A4 < A5
# Index-based (1-based): c(2, 3, 1, 4, 5)
ranking_order <- c(2, 3, 1, 4, 5)
ac_res <- ac.ranking(R, ranking_order)
results$ac_ranking_mu <- as.vector(ac_res)

# 3. Turnover Constraint
# Base optimization to get w_init
pspec <- portfolio.spec(assets=colnames(R))
pspec <- add.constraint(portfolio=pspec, type="full_investment")
pspec <- add.constraint(portfolio=pspec, type="box", min=0, max=1)
pspec <- add.objective(portfolio=pspec, type="risk", name="StdDev")

# Optimize without turnover
opt_base <- optimize.portfolio(R=R, portfolio=pspec, optimize_method="ROI")
w_init <- opt_base$weights

# Now add turnover constraint (target = 0.2)
pspec_to <- add.constraint(portfolio=pspec, type="turnover", turnover_target=0.2, weight_initial=w_init)
# Use CVXR to solve if ROI doesn't handle turnover nicely, but ROI with CVXR/glpk should work
# PortfolioAnalytics' ROI solver for turnover uses CVXR if available
opt_to <- optimize.portfolio(R=R, portfolio=pspec_to, optimize_method="ROI")

results$turnover_w_init <- as.vector(w_init)
results$turnover_target <- 0.2
results$turnover_weights <- as.vector(opt_to$weights)

# Save to JSON
write_json(results, "data/new_features_cv.json", digits=10)
print("Generated data/new_features_cv.json")
