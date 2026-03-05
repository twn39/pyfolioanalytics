using DelimitedFiles, LinearAlgebra, JuMP, Ipopt # Ipopt is usually available in standard julia environments or common for NLP/LP

# Zero-dependency OWA logic from PortfolioOptimisers.jl source
function owa_gmd(T::Integer)
    return (4 * (1:T) .- 2 * (T + 1)) / (T * (T - 1))
end

# Load data - EDHEC uses ';' and '%' strings
csv_data = readdlm("data/edhec.csv", ';', skipstart=1)
R = convert(Matrix{Float64}, [parse(Float64, endswith(string(val), "%") ? string(val)[1:end-1] : string(val)) / (endswith(string(val), "%") ? 100.0 : 1.0) for val in csv_data[:, 2:11]])
T, N = size(R)

# GMD Weights (increasing as per PO.jl)
w_gmd = owa_gmd(T)

# Solve Min OWA (GMD) using JuMP
model = Model(Ipopt.Optimizer)
set_silent(model)
@variable(model, w[1:N] >= 0)
@constraint(model, sum(w) == 1.0)

# OWA math in Julia (using sorting formulation)
# OWA = dot(w_gmd, sort(-R*w))
# In optimization, we need the LP formulation for sorted gains/losses
@variable(model, losses[1:T])
@constraint(model, losses .== -R * w)

# OWA formulation for increasing weights w_gmd and INCREASINGLY sorted losses
# PO.jl formulation: OWA = dot(w_gmd, sorted_losses_inc)
# This is equivalent to our py formulation if we flip signs/order.
# LP: min sum_{k=1}^{T-1} (w_k - w_{k+1}) * (sum of k smallest losses) + ...

# For simplicity, let's use the exact weight generation and risk calculation for comparison first.
# If JuMP is missing, this will fail. Let's try to do it without an optimizer first if it's too risky.
# Actually, the user wants REAL data cross-val.

@variable(model, zeta[1:T-1])
@variable(model, d[1:T, 1:T-1] >= 0)
for k in 1:T-1
    # Sum of k smallest losses = max k*zeta_k - sum(d_{t,k})
    # losses_t >= zeta_k - d_{t,k}
    @constraint(model, losses .>= zeta[k] .- d[:, k])
end

# w_gmd is increasing: w_k - w_{k+1} < 0
delta_w = w_gmd[1:end-1] .- w_gmd[2:end]
@objective(model, Min, sum(delta_w[k] * (k * zeta[k] - sum(d[:, k])) for k in 1:T-1) + w_gmd[end] * sum(losses))

optimize!(model)

opt_w = value.(w)
opt_risk = objective_value(model)

println("---BEGIN RESULTS---")
println("T: $T")
println("N: $N")
println("opt_risk_gmd: $opt_risk")
println("opt_weights: ", opt_w)
println("---END RESULTS---")
