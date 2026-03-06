using DelimitedFiles, LinearAlgebra, Statistics

# Load data - EDHEC standard CSV
csv_data = readdlm("data/edhec.csv", ',', skipstart=1)
# Use all 13 assets (cols 2:14) and all rows
R = convert(Matrix{Float64}, csv_data[:, 2:14])
T, N = size(R)
q = T / N

# Original Covariance
sigma_raw = cov(R)
# Convert to Correlation
std_devs = sqrt.(diag(sigma_raw))
corr = sigma_raw ./ (std_devs * std_devs')

# Eigen-decomposition
vals, vecs = eigen(corr)
# Sort descending
idx = sortperm(vals, rev=true)
vals = vals[idx]
vecs = vecs[:, idx]

# Fixed threshold for cross-validation parity
e_max = (1 + sqrt(1.0 / q))^2
n_noise = sum(vals .<= e_max)

vals_denoised = copy(vals)
avg_noise = mean(vals[end-n_noise+1:end])
vals_denoised[end-n_noise+1:end] .= avg_noise

corr_denoised = vecs * Diagonal(vals_denoised) * vecs'
# Ensure unit diagonal
d = diag(corr_denoised)
corr_denoised = corr_denoised ./ sqrt.(d * d')

sigma_denoised = corr_denoised .* (std_devs * std_devs')

# Manually write JSON to avoid dependency issues
function write_json_manual(filename, data)
    open(filename, "w") do f
        write(f, "{\n")
        write(f, "  \"q\": $(data["q"]),\n")
        write(f, "  \"e_max\": $(data["e_max"]),\n")
        
        # Matrix to nested list
        function mat_to_str(M)
            rows = []
            for i in 1:size(M, 1)
                push!(rows, "[" * join(M[i, :], ", ") * "]")
            end
            return "[" * join(rows, ", ") * "]"
        end
        
        write(f, "  \"sigma_raw\": $(mat_to_str(data["sigma_raw"])),\n")
        write(f, "  \"sigma_denoised\": $(mat_to_str(data["sigma_denoised"]))\n")
        write(f, "}\n")
    end
end

output_data = Dict(
    "q" => q,
    "e_max" => e_max,
    "sigma_raw" => sigma_raw,
    "sigma_denoised" => sigma_denoised
)

write_json_manual("data/rmt_cv.json", output_data)
println("Generated data/rmt_cv.json (manual JSON)")
