function solve_portfolio(past_returns, Gamma, pref_cuts)

# Create covariance matrix and mean vector
covar = cov(past_returns)
means = mean(past_returns, 1)

# Idea: multivariate normals can be described as
# r = A * z + mu
# where A*A^T = covariance matrix.
# Instead of building uncertainty set limiting variation of r
# directly, we constrain the "independent" z
A = round(chol(covar),2)

# Setup the robust optimization model
m = RobustModel(solver=GurobiSolver(OutputFlag=0))

# Variables
@defVar(m, obj)  # Put objective as constraint using dummy variable
@defVar(m, x[1:NUM_ASSET] >= 0)
# Uncertainties
@defUnc(m,       r[1:NUM_ASSET]      )  # The returns
@defUnc(m, -1 <= z[1:NUM_ASSET] <= 1 )  # The "standard normals"
@defUnc(m,  0 <= y[1:NUM_ASSET] <= 1 )  # |z|/box

@setObjective(m, Max, obj)

# Portfolio constraint
addConstraint(m, sum([ x[i] for i=1:NUM_ASSET ]) == 1)

# The objective constraint - uncertain
addConstraint(m, sum([ r[i]*x[i] for i=1:NUM_ASSET ]) - obj >= 0)

# Build uncertainty set
# First, link returns to the standard normals
for asset_ind = 1:NUM_ASSET
addConstraint(m, r[asset_ind] ==
sum([ A[asset_ind, j] * z[j] for j=1:NUM_ASSET ]) + means[asset_ind] )
end
# Then link absolute values to standard normals
for asset_ind = 1:NUM_ASSET
addConstraint(m, y[asset_ind] >= -z[asset_ind] / box)
addConstraint(m, y[asset_ind] >=  z[asset_ind] / box)
end
# Finally, limit how much the standard normals can vary from means
addConstraint(m, sum([ y[j] for j=1:NUM_ASSET ]) <= Gamma)

solveRobust(m, prefer_cuts=pref_cuts)

return getValue(x)

end  # end of function

function generate_data(num_samples)
data = zeros(N, NUM_ASSET)

# Linking factors
beta = [(i-1.)/NUM_ASSET for i = 1:NUM_ASSET]

for sample_ind = 1:num_samples
# Common market factor, mean 3%, sd 5%, truncate at +- 3 sd
z = rand(Normal(0.03, 0.05))
z = max(z, 0.03 - 3*0.05)
z = min(z, 0.03 + 3*0.05)

for asset_ind = 1:NUM_ASSET
# Idiosyncratic contribution, mean 0%, sd 5%, truncated at +- 3 sd
asset = rand(Normal(0.00, 0.05))
asset = max(asset, 0.00 - 3*0.05)
asset = min(asset, 0.00 + 3*0.05)
data[sample_ind, asset_ind] = beta[asset_ind] * z + asset
end
end

return data
end

past_returns   = generate_data(1000)
future_returns = generate_data(1000)

function eval_gamma(Gamma)
x = solve_portfolio(past_returns, 1, true)
future_z = future_returns * x[:]
sort!(future_z)
println("Selected solution summary stats for Gamma $Gamma")
println("10%:     ", future_z[int(NUM_FUTURE*0.1)])
println("20%:     ", future_z[int(NUM_FUTURE*0.2)])
println("30%:     ", future_z[int(NUM_FUTURE*0.3)])
println("Mean:    ", mean(future_z))
println("Maximum: ", future_z[end])
end

eval_gamma(0)  # Nominal - no uncertainty
eval_gamma(3)  # Some protection