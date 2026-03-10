library(PortfolioAnalytics)
library(jsonlite)

# Load data
data(edhec)
R <- as.matrix(edhec[, 1:6])
T <- nrow(R)
N <- ncol(R)
asset_names <- colnames(R)

# Define Factor Loading Matrix B (N x 2)
# Let's create dummy factors: 
# Factor 1: Average return of the first 3 assets
# Factor 2: Variance of each asset (normalized)
B <- matrix(0, nrow=N, ncol=2)
B[, 1] <- c(0.8, 0.9, 1.1, 0.5, 0.4, 0.3)
B[, 2] <- c(0.2, 0.1, -0.1, 0.4, 0.5, 0.6)
colnames(B) <- c("Growth", "Value")
rownames(B) <- asset_names

# Portfolio Specification
pspec <- portfolio.spec(assets=asset_names)
pspec <- add.constraint(portfolio=pspec, type="full_investment")
pspec <- add.constraint(portfolio=pspec, type="box", min=0, max=0.4)

# Add Factor Exposure Constraint
# Growth exposure between 0.5 and 0.7
# Value exposure between 0.1 and 0.3
lower <- c(0.5, 0.1)
upper <- c(0.7, 0.3)
pspec <- add.constraint(portfolio=pspec, type="factor_exposure", B=B, lower=lower, upper=upper)

# Optimization: Maximize Return (to make constraints binding)
pspec <- add.objective(portfolio=pspec, type="return", name="mean")

# Solve using ROI (glpk) for linear problem
# Factor exposure in ROI is supported
res <- optimize.portfolio(R=R, portfolio=pspec, optimize_method="ROI")

# Export
output <- list(
  input_R = R,
  B = B,
  lower = lower,
  upper = upper,
  weights = as.vector(res$weights),
  exposure_actual = as.vector(t(B) %*% res$weights)
)

write_json(output, "data/factor_exposure_cv.json", digits=10)
