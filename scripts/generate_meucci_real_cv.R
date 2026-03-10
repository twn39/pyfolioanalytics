library(PortfolioAnalytics)
library(jsonlite)

data(edhec)
R <- as.matrix(edhec)
T <- nrow(R)
N <- ncol(R)
p_prior <- rep(1/T, T)

asset_names <- colnames(R)
idx_ca <- which(asset_names == "Convertible Arbitrage")
idx_cta <- which(asset_names == "CTA Global")
idx_ds <- which(asset_names == "Distressed Securities")

# Constructing Aeq
# Aeq should be K x T
Aeq <- matrix(0, nrow=2, ncol=T)
Aeq[1, ] <- 1
Aeq[2, ] <- R[, idx_ca]

# beq should be K x 1
beq <- matrix(c(1, 0.004), nrow=2)

# Constructing Aineq
# Aineq should be K_ x T
Aineq <- matrix(0, nrow=1, ncol=T)
Aineq[1, ] <- R[, idx_ds] - R[, idx_cta]
bineq <- matrix(0, nrow=1)

# Run EntropyProg
res_mixed <- EntropyProg(p_prior, A=Aineq, b=bineq, Aeq=Aeq, beq=beq)

# Export results
output <- list(
  input_R = R,
  prior_probs = p_prior,
  mixed_views = list(
    p_posterior = as.vector(res_mixed$p_),
    mu = as.vector(meucci.moments(R, res_mixed$p_)$mu),
    sigma = as.matrix(meucci.moments(R, res_mixed$p_)$sigma)
  )
)

write_json(output, "data/meucci_real_cv.json", digits=10)
