library(PortfolioAnalytics)
library(fGarch)
library(jsonlite)

# Load data
data(edhec)
# Use a subset of assets and rows for faster and more stable GARCH fitting
R <- as.matrix(edhec[1:100, 1:3])
colnames(R) <- c("CA", "CTAG", "DS")

# Calculate CCC GARCH Moments using the function from PA
res_garch <- CCCgarch.MM(R)

# Export results
output <- list(
  input_R = R,
  mu = as.vector(res_garch$mu),
  sigma = as.matrix(res_garch$sigma),
  m3 = as.matrix(res_garch$m3),
  m4 = as.matrix(res_garch$m4)
)

write_json(output, "data/garch_cv.json", digits=10)
