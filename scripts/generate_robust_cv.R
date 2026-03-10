library(robustbase)
library(jsonlite)

set.seed(42)
T <- 100
N <- 5
R <- matrix(rnorm(T * N, mean = 0, sd = 0.01), nrow = T, ncol = N)

# Compute MCD covariance
# Using same default alpha as sklearn: (n+p+1)/2 / n
# Sklearn: floor((n_samples + n_features + 1) / 2) / n_samples
alpha_val <- floor((T + N + 1) / 2) / T

mcd_res <- covMcd(R, alpha = alpha_val)

results <- list(
  T = T,
  N = N,
  returns = R,
  mu = as.numeric(mcd_res$center),
  sigma = mcd_res$cov
)

write_json(results, "data/robust_cv.json", auto_unbox = TRUE, pretty = TRUE)
