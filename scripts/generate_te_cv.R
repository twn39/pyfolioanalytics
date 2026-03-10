library(PortfolioAnalytics)
library(jsonlite)
library(xts)
library(CVXR)

set.seed(42)
n_assets <- 5
n_obs <- 100
dates <- seq(as.Date("2020-01-01"), length.out=n_obs, by="day")
R_mat <- matrix(rnorm(n_obs * n_assets), nrow=n_obs, ncol=n_assets)
colnames(R_mat) <- paste0("A", 1:n_assets)
R <- xts(R_mat, order.by=dates)

mu <- as.vector(apply(R, 2, "mean"))
sigma <- cov(R)

# Benchmark weights: say equal weight in first 3 assets
w_b <- c(0.33, 0.33, 0.34, 0, 0)
names(w_b) <- colnames(R)

# Tracking error target: 2% (0.02)
te_target <- 0.02

# Manual CVXR for ground truth parity
w <- Variable(n_assets)
obj <- Minimize(quad_form(w, sigma)) # Min variance s.t. TE constraint
# TE constraint: sqrt((w - w_b)' * sigma * (w - w_b)) <= target
# (w - w_b)' * sigma * (w - w_b) <= target^2
constraints <- list(
  sum(w) == 1,
  w >= 0,
  quad_form(w - w_b, sigma) <= te_target^2
)
prob <- Problem(obj, constraints)
res <- solve(prob)

opt_weights <- as.numeric(res$getValue(w))
names(opt_weights) <- colnames(R)

results <- list()
results$returns <- as.matrix(R)
results$benchmark_weights <- w_b
results$te_target <- te_target
results$opt_weights <- opt_weights
results$te_actual <- as.numeric(sqrt(t(opt_weights - w_b) %*% sigma %*% (opt_weights - w_b)))

write_json(results, "data/te_cv.json", digits=10)
print("Generated data/te_cv.json")
