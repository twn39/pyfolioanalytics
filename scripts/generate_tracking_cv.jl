using DelimitedFiles, LinearAlgebra, JuMP, Ipopt
using JSON, Statistics

# Load data - subset
csv_data = readdlm("data/stock_returns.csv", ',', skipstart=1)
R = convert(Matrix{Float64}, csv_data[1:100, 2:6])
T, N = size(R)

mu = vec(mean(R, dims=1))
Sigma = cov(R)

w_bench = fill(1.0 / N, N)
target_L_inf = 0.05
target_L_1 = 0.10

# --- Model 1: L-infinity Tracking Error ---
model_inf = Model(Ipopt.Optimizer)
set_silent(model_inf)
@variable(model_inf, w_inf[1:N])
@constraint(model_inf, sum(w_inf) == 1.0)
@constraint(model_inf, w_inf .>= 0)
@constraint(model_inf, w_inf .- w_bench .<= target_L_inf)
@constraint(model_inf, w_inf .- w_bench .>= -target_L_inf)

# Minimize Variance
@objective(model_inf, Min, w_inf' * Sigma * w_inf)
optimize!(model_inf)
w_inf_opt = value.(w_inf)

# --- Model 2: L-1 Tracking Error ---
model_1 = Model(Ipopt.Optimizer)
set_silent(model_1)
@variable(model_1, w_1[1:N] >= 0)
@variable(model_1, dev_1[1:N] >= 0)
@constraint(model_1, sum(w_1) == 1.0)
@constraint(model_1, dev_1 .>= w_1 .- w_bench)
@constraint(model_1, dev_1 .>= -(w_1 .- w_bench))
@constraint(model_1, sum(dev_1) <= target_L_1)

@objective(model_1, Min, w_1' * Sigma * w_1)
optimize!(model_1)
w_1_opt = value.(w_1)

results = Dict(
    "L_inf_weights" => w_inf_opt,
    "L_1_weights" => w_1_opt
)

open("data/tracking_cv.json", "w") do f
    write(f, JSON.json(results, 4))
end

println("Successfully generated data/tracking_cv.json")
