library(PortfolioAnalytics)
library(jsonlite)
library(xts)

# Helper for EVaR parity with Python implementation
# EVaR_p(X) = inf_{z>0} { z * ln( E[exp(X/z)] / (1-p) ) } where X is loss (-returns)
calc_evar <- function(R, weights, p = 0.95) {
  R_mat <- as.matrix(R)
  X <- -(R_mat %*% weights) # Losses
  alpha <- 1 - p
  T_len <- length(X)
  
  obj <- function(z) {
    if (z <= 0) return(1e10)
    # Numerical stability: use log-sum-exp trick
    m <- max(X/z)
    val <- z * (m + log(sum(exp(X/z - m)) / (T_len * alpha)))
    return(val)
  }
  
  res <- optimize(obj, interval = c(1e-6, 100))
  return(res$objective)
}

# Preprocessing functions matching Python load_dataset
load_edhec <- function() {
  df <- read.table("data/edhec.csv", sep=";", header=TRUE, check.names=FALSE)
  dates <- as.Date(df[,1], format="%d/%m/%Y")
  df_data <- df[,-1]
  
  # Match column naming: space to dot
  colnames(df_data) <- gsub(" ", ".", colnames(df_data))
  
  # Handle percentage strings
  for(i in 1:ncol(df_data)) {
    if(is.character(df_data[,i])) {
      df_data[,i] <- as.numeric(gsub("%", "", df_data[,i])) / 100
    }
  }
  
  # Select first 5 assets as in Python test
  res <- xts(df_data[, 1:5], order.by=dates)
  return(res)
}

load_stocks <- function() {
  df <- read.csv("data/stock_returns.csv", row.names=1)
  dates <- as.Date(rownames(df))
  res <- xts(df, order.by=dates)
  return(res)
}

load_macro <- function() {
  df <- read.csv("data/macro_returns.csv", row.names=1)
  dates <- as.Date(rownames(df))
  # Match dash-to-dot column naming
  colnames(df) <- gsub("-", ".", colnames(df))
  res <- xts(df, order.by=dates)
  return(res)
}

# Main validation logic
datasets <- list(
  edhec = load_edhec(),
  stocks = load_stocks(),
  macro = load_macro()
)

results <- list()

for(name in names(datasets)) {
  R <- datasets[[name]]
  assets <- colnames(R)
  
  # 1. Base MVO
  port_mvo <- portfolio.spec(assets = assets)
  port_mvo <- add.constraint(portfolio = port_mvo, type = "full_investment")
  port_mvo <- add.constraint(portfolio = port_mvo, type = "long_only")
  port_mvo <- add.objective(portfolio = port_mvo, type = "risk", name = "StdDev")
  
  # Use ROI for parity
  opt_mvo <- optimize.portfolio(R = R, portfolio = port_mvo, optimize_method = "ROI")
  mvo_weights <- as.numeric(extractWeights(opt_mvo))
  
  # 2. EVaR (Equal Weight)
  n_assets <- length(assets)
  eq_weights <- rep(1/n_assets, n_assets)
  evar_eq <- calc_evar(R, eq_weights, p = 0.95)
  
  # 3. Robust MVO (Utility objective with adjusted mu)
  # Python uses delta_mu=0.0001, risk_aversion=2.0
  mu <- colMeans(R)
  sigma <- cov(R)
  mu_robust <- mu - 0.0001
  
  port_rob <- portfolio.spec(assets = assets)
  port_rob <- add.constraint(portfolio = port_rob, type = "full_investment")
  port_rob <- add.constraint(portfolio = port_rob, type = "long_only")
  # quadratic_utility in PortfolioAnalytics: mu'w - 0.5 * risk_aversion * w'Sigma w
  # Python's solve_mvo uses 0.5 * risk_aversion * w'Sigma w
  # In PortfolioAnalytics, the risk_aversion in add.objective(type='risk', name='var')
  # is exactly the lambda in lambda * w'Sigma w.
  # To match Python's 0.5 * 2.0 = 1.0, we use 1.0 here.
  port_rob <- add.objective(portfolio = port_rob, type = "return", name = "mean")
  port_rob <- add.objective(portfolio = port_rob, type = "risk", name = "var", risk_aversion = 1.0)
  
  # Pass custom moments to optimize.portfolio
  # PortfolioAnalytics ROI solver expects 'mean' and 'var'
  opt_rob <- optimize.portfolio(R = R, portfolio = port_rob, optimize_method = "ROI", 
                                 momentargs = list(mean = mu_robust, var = sigma))
  rob_weights <- as.numeric(extractWeights(opt_rob))
  
  results[[name]] <- list(
    assets = assets,
    mvo_weights = mvo_weights,
    evar_eq = evar_eq,
    robust_weights = rob_weights
  )
}

write_json(results, "data/multi_cross_val.json", auto_unbox = TRUE, digits = 10)
cat("Successfully generated data/multi_cross_val.json\n")
