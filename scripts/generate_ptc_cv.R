library(PortfolioAnalytics)
library(jsonlite)
library(xts)

set.seed(42)
n_assets <- 5
n_obs <- 50
dates <- seq(as.Date("2020-01-01"), length.out=n_obs, by="day")
R_mat <- matrix(rnorm(n_obs * n_assets), nrow=n_obs, ncol=n_assets)
colnames(R_mat) <- paste0("A", 1:n_assets)
R <- xts(R_mat, order.by=dates)

# Initial weights for turnover/PTC
w_init <- rep(1/n_assets, n_assets)
names(w_init) <- colnames(R)

# 1. Proportional Transaction Costs (PTC)
# In PortfolioAnalytics, PTC can be added as a constraint type="transaction_cost"
# with ptc argument.
pspec <- portfolio.spec(assets=colnames(R))
pspec <- add.constraint(portfolio=pspec, type="full_investment")
pspec <- add.constraint(portfolio=pspec, type="box", min=0, max=1)
# Add PTC as an objective penalty instead of constraint to ensure it's handled by ROI
# PortfolioAnalytics' turnover objective can act as a transaction cost penalty.
ptc_val <- 0.01
pspec_ptc <- add.objective(portfolio=pspec, type="turnover", name="turnover", 
                          turnover_target=NULL, weight_initial=w_init, 
                          multiplier=ptc_val)

# We need objectives for optimization
pspec_ptc <- add.objective(portfolio=pspec_ptc, type="return", name="mean")
# Note: risk_aversion=2 in R means 0.5 * 2 * w'Sw = 1.0 * w'Sw
pspec_ptc <- add.objective(portfolio=pspec_ptc, type="risk", name="var", risk_aversion=2)

library(CVXR)
# Extract moments
mu <- as.vector(apply(R, 2, "mean"))
sigma <- cov(R)
n <- n_assets

# CVXR optimization
w <- Variable(n)
risk_aversion <- 2
ptc_val <- 0.01

obj <- Minimize(0.5 * risk_aversion * quad_form(w, sigma) - t(mu) %*% w + sum(abs(w - w_init)) * ptc_val)
constraints <- list(sum(w) == 1, w >= 0)
prob <- Problem(obj, constraints)
res <- solve(prob)

opt_weights <- as.numeric(res$getValue(w))
names(opt_weights) <- colnames(R)

results <- list()
results$returns <- as.matrix(R)
results$w_init <- as.vector(w_init)
results$ptc <- ptc_val
results$opt_weights <- opt_weights
# Manually construct objective measures for parity check
results$objective_measures <- list(
  mean = as.numeric(t(opt_weights) %*% mu),
  StdDev = as.numeric(sqrt(t(opt_weights) %*% sigma %*% opt_weights))
)

# Save to JSON
write_json(results, "data/ptc_cv.json", digits=10)
print("Generated data/ptc_cv.json")
