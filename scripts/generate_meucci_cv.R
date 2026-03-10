library(PortfolioAnalytics)
library(jsonlite)

# Load data
data(edhec)
R <- as.matrix(edhec[, 1:5])
T <- nrow(R)
N <- ncol(R)

# 1. Test Basic EntropyProg with Equality Constraints
p_prior <- rep(1/T, T)
# View: Expected return of first asset is 0.005 AND sum(p) = 1
Aeq <- rbind(rep(1, T), R[, 1])
beq <- rbind(1.0, 0.005)
res_eq <- EntropyProg(p_prior, Aeq=Aeq, beq=beq)

# 2. Test EntropyProg with Inequality Constraints
# View: Expected return of second asset > third asset AND sum(p) = 1
Aineq <- matrix(R[, 3] - R[, 2], nrow=1)
bineq <- matrix(0, nrow=1)
Aeq_ones <- matrix(rep(1, T), nrow=1)
beq_ones <- matrix(1, nrow=1)
res_ineq <- EntropyProg(p_prior, A=Aineq, b=bineq, Aeq=Aeq_ones, beq=beq_ones)

# 3. Test meucci.ranking
# Order: asset 2 < asset 3 < asset 1 < asset 4 < asset 5
order_vec <- c(2, 3, 1, 4, 5)
res_ranking <- meucci.ranking(R, p_prior, order_vec)

# Export to JSON
output <- list(
  input_R = R, # Export full R for parity
  prior_probs = p_prior,
  entropy_prog_eq = list(
    p_posterior = as.vector(res_eq$p_),
    converged = res_eq$optimizationPerformance$converged
  ),
  entropy_prog_ineq = list(
    p_posterior = as.vector(res_ineq$p_),
    converged = res_ineq$optimizationPerformance$converged
  ),
  meucci_ranking = list(
    mu = as.vector(res_ranking$mu),
    sigma = as.matrix(res_ranking$sigma)
  )
)

write_json(output, "data/meucci_cv.json", digits=10)
