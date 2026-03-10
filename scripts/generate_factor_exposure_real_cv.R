library(PortfolioAnalytics)
library(jsonlite)

# Load real data
data(edhec)
R <- as.matrix(edhec)
T <- nrow(R)
N <- ncol(R)
asset_names <- colnames(R)

# Define a more complex Factor Loading Matrix B (N x 3)
# Factors: Market, Arbitrage, Macro
B <- matrix(0, nrow=N, ncol=3)
colnames(B) <- c("Market", "Arbitrage", "Macro")
rownames(B) <- asset_names

# Dummy loadings for real assets
B[, "Market"] <- c(0.4, 0.1, 0.2, 0.8, 0.05, 0.3, 0.1, 0.6, 0.9, 0.2, 0.1, -0.2, 0.4)
B[, "Arbitrage"] <- c(0.9, 0.05, 0.8, 0.1, 0.7, 0.6, 0.9, 0.1, 0.1, 0.8, 0.7, 0.0, 0.5)
B[, "Macro"] <- c(0.1, 0.8, 0.1, 0.2, 0.3, 0.4, 0.1, 0.9, 0.2, 0.1, 0.2, 0.8, 0.6)

# Portfolio Specification
pspec <- portfolio.spec(assets=asset_names)
pspec <- add.constraint(portfolio=pspec, type="full_investment")
pspec <- add.constraint(portfolio=pspec, type="box", min=0, max=0.2) # Max 20% per asset

# Add Complex Factor Exposure Constraints
# Market beta between 0.3 and 0.5
# Arbitrage exposure > 0.4
# Macro exposure < 0.3
lower <- c(0.3, 0.4, -1.0)
upper <- c(0.5, 1.0, 0.3)
pspec <- add.constraint(portfolio=pspec, type="factor_exposure", B=B, lower=lower, upper=upper)

# Objective: Minimize Variance (Quadratic Problem)
pspec <- add.objective(portfolio=pspec, type="risk", name="var")

# Solve using ROI (quadprog)
res <- optimize.portfolio(R=R, portfolio=pspec, optimize_method="ROI")

# Export results
output <- list(
  input_R = R,
  B = B,
  lower = lower,
  upper = upper,
  weights = as.vector(res$weights),
  exposure_actual = as.vector(t(B) %*% res$weights),
  status = "optimal"
)

write_json(output, "data/factor_exposure_real_cv.json", digits=10)
