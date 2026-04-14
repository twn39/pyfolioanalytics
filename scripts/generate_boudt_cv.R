library(PerformanceAnalytics)
library(jsonlite)
library(robustbase)

df <- read.csv("data/stock_returns.csv", row.names=1)
R <- df[1:100, 1:5]
R[10, 1] <- R[10, 1] * 10 + 0.5
R[50, 3] <- R[50, 3] * 10 - 0.5

# PerformanceAnalytics calls `Return.clean` -> `Return.clean.boudt` -> `robustbase::covMcd`
# Wait, Return.clean uses `covMcd(R, alpha=1-alpha)` where alpha is the quantile.
cleanU <- Return.clean(R, method="boudt", alpha=0.05)

write_json(list(
    cleaned_returns = as.matrix(cleanU),
    original_returns = as.matrix(R)
), "data/boudt_cv.json", pretty=TRUE)
