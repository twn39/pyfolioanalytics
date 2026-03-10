library(PortfolioAnalytics)
library(jsonlite)
library(nloptr)

# Setup test data
set.seed(42)
T <- 100
N <- 3
R_raw <- matrix(rnorm(T * N, mean = 0, sd = 0.01), nrow = T, ncol = N)
colnames(R_raw) <- c("A", "B", "C")
p_prior <- rep(1/T, T)

# 1. Test meucci.ranking (C < B < A)
order_ranking <- c(3, 2, 1)
res_ranking <- meucci.ranking(R_raw, p_prior, order_ranking)
# R's meucci.ranking returns a list(mu=..., sigma=...) but it also computes p internally.
# I need to capture p_ from the internal call or use meucci.moments manually.

# 2. Test Absolute View (A = 0.01)
Aeq <- rbind(rep(1, T), R_raw[, 1])
beq <- c(1.0, 0.01)
res_absolute <- EntropyProg(p_prior, Aeq = Aeq, beq = beq)

print("Ranking Result structure:")
print(names(res_ranking))
print("Absolute Result structure:")
print(names(res_absolute))

results <- list(
  T = T,
  N = N,
  returns = R_raw,
  p_prior = as.numeric(p_prior),
  # Ranking results
  ranking_order = order_ranking,
  mu_ranking = as.numeric(res_ranking$mu),
  # Absolute view results
  target_mu_A = 0.01,
  p_absolute = as.numeric(res_absolute$p_),
  mu_absolute = as.numeric(colSums(R_raw * as.numeric(res_absolute$p_)))
)

print(str(results))

write_json(results, "data/meucci_cv.json", auto_unbox = TRUE, pretty = TRUE, digits = 10)
