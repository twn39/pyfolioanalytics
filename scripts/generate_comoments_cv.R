library(PortfolioAnalytics)
library(PerformanceAnalytics)
library(jsonlite)
library(xts)

# Setup test data
set.seed(42)
T <- 50
N <- 4
R_raw <- matrix(rnorm(T * N, mean = 0, sd = 0.01), nrow = T, ncol = N)
colnames(R_raw) <- c("A", "B", "C", "D")

R_xts <- xts(R_raw, order.by = as.Date(1:T, origin = "2020-01-01"))

# 1. Sample Moments
m3_sample <- PerformanceAnalytics::M3.MM(R_xts)
m4_sample <- PerformanceAnalytics::M4.MM(R_xts)

# 2. Factor Model Moments (k=1)
fm1 <- statistical.factor.model(R_xts, k = 1)
m3_fm1 <- extractCoskewness(fm1)
m4_fm1 <- extractCokurtosis(fm1)

# 3. Factor Model Moments (k=2)
fm2 <- statistical.factor.model(R_xts, k = 2)
m3_fm2 <- extractCoskewness(fm2)
m4_fm2 <- extractCokurtosis(fm2)

results <- list(
  T = T,
  N = N,
  returns = R_raw,
  # Sample
  m3_sample = as.numeric(m3_sample),
  m4_sample = as.numeric(m4_sample),
  # FM k=1
  m3_fm1 = as.numeric(m3_fm1),
  m4_fm1 = as.numeric(m4_fm1),
  # FM k=2
  m3_fm2 = as.numeric(m3_fm2),
  m4_fm2 = as.numeric(m4_fm2)
)

write_json(results, "data/comoments_cv.json", auto_unbox = TRUE, pretty = TRUE, digits = 10)
