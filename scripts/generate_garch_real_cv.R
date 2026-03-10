library(PortfolioAnalytics)
library(fGarch)
library(jsonlite)

# Load full real data
data(edhec)
R <- as.matrix(edhec)
T <- nrow(R)
N <- ncol(R)

# Calculate CCC GARCH Moments for all 13 assets
# Using the standard PA function
res_garch <- CCCgarch.MM(R)

# Export results
output <- list(
  dataset = "edhec_full",
  input_R = R,
  mu = as.vector(res_garch$mu),
  sigma = as.matrix(res_garch$sigma),
  # M3 and M4 are large (N x N^2 and N x N^3), so we'll export 
  # a subset or summary stats if they are too big, but for 13 assets 
  # N=13, N^2=169, N^3=2197. Total elements ~ 30k. JSON should handle it.
  m3 = as.matrix(res_garch$m3),
  m4 = as.matrix(res_garch$m4)
)

write_json(output, "data/garch_real_cv.json", digits=10)
