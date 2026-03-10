library(PerformanceAnalytics)
library(jsonlite)

data(edhec)
R <- as.matrix(edhec[, 1:5])
N <- ncol(R)
weights <- rep(1/N, N)

# Compute Covariance
sigma <- cov(R)
total_sd <- as.numeric(sqrt(t(weights) %*% sigma %*% weights))

# Component Contribution using Euler
# CCR_i = w_i * (Sigma * w)_i / total_sd
ccr_sd <- as.vector(weights * (sigma %*% weights) / total_sd)
mcr_sd <- as.vector((sigma %*% weights) / total_sd)
pcr_sd <- ccr_sd / total_sd

# Total Var
total_var <- as.numeric(t(weights) %*% sigma %*% weights)
ccr_var <- as.vector(weights * (sigma %*% weights))
pcr_var <- ccr_var / total_var

# Export results
output <- list(
  sigma = sigma,
  weights = weights,
  std_dev_decomp = list(
    total = total_sd,
    mcr = mcr_sd,
    ccr = ccr_sd,
    pcr = pcr_sd
  ),
  var_decomp = list(
    total = total_var,
    ccr = ccr_var,
    pcr = pcr_var
  )
)

write_json(output, "data/risk_decomposition_cv.json", digits=10)
