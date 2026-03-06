library(PortfolioAnalytics)
library(jsonlite)

# Load data
data(edhec)
# Use first 5 assets and first 24 months
edhec_subset <- edhec[1:24, 1:5]
asset_names <- colnames(edhec_subset)

# Define portfolio
pspec <- portfolio.spec(assets=asset_names)
pspec <- add.constraint(portfolio=pspec, type="full_investment")
pspec <- add.constraint(portfolio=pspec, type="box", min=0, max=1)
pspec <- add.objective(portfolio=pspec, type="risk", name="StdDev")
pspec <- add.objective(portfolio=pspec, type="return", name="mean")

# Calculate efficient frontier using ROI
ef <- create.EfficientFrontier(R=edhec_subset, portfolio=pspec, type="mean-StdDev", n.portfolios=20)

# The column names for weights are paste0("w.", asset_names)
weight_cols <- paste0("w.", asset_names)
weights_mat <- ef$frontier[, weight_cols]

# Prepare output with full precision
output <- list(
  mu = as.numeric(colMeans(edhec_subset)),
  sigma = as.matrix(cov(edhec_subset)),
  frontier_weights = as.matrix(weights_mat),
  frontier_means = as.numeric(ef$frontier[, "mean"]),
  frontier_stds = as.numeric(ef$frontier[, "StdDev"])
)

# Use rowmajor matrix format
write_json(output, "data/cla_cv.json", matrix="rowmajor", digits=NA)
cat("Generated data/cla_cv.json with correct weight column names\n")
