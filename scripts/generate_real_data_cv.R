library(PortfolioAnalytics)
library(jsonlite)
library(xts)

# Load EDHEC data
edhec_df <- read.csv("data/edhec.csv", check.names = FALSE)
# Convert to xts
# Date is in DD/MM/YYYY
dates <- as.Date(edhec_df$date, format="%d/%m/%Y")
R_xts <- xts(edhec_df[,-1], order.by=dates)

# Select subset of assets to keep computation fast but representative
R_sub <- R_xts[, 1:5]

# 1. Robust Moments (MCD)
# PortfolioAnalytics uses robustbase::covMcd
mcd_res <- robustbase::covMcd(R_sub, alpha = floor((nrow(R_sub) + ncol(R_sub) + 1) / 2) / nrow(R_sub))

# 2. Factor Model Moments (k=1)
fm1 <- statistical.factor.model(R_sub, k = 1)
m3_fm1 <- extractCoskewness(fm1)
m4_fm1 <- extractCokurtosis(fm1)

results <- list(
  asset_names = colnames(R_sub),
  # Robust
  mu_robust = as.numeric(mcd_res$center),
  sigma_robust = mcd_res$cov,
  # FM k=1
  m3_fm1 = as.numeric(m3_fm1),
  m4_fm1 = as.numeric(m4_fm1)
)

write_json(results, "data/real_data_cv.json", auto_unbox = TRUE, pretty = TRUE, digits = 10)
