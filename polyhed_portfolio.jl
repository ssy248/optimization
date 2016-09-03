# max   obj
# s.t.  sum(x_i      for i in 1:n) == 1
#       sum(r_i x_i  for i in 1:n) >= obj
#       x_i >= 0
# where x_i is the percent allocation for asset i, and r_i is the
# uncertain return on asset i.
#
# Uncertainty set:
#       r = A z + mean
#       y = |z|
#       sum(y_i for i in 1:n) <= Gamma
#       |z| <= 1, 0 <= y <= 1
# where A is such that A*A' = Covariance matrix


function generate_data(num_samples)
    N = 10
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

# gamma is size of uncertainty set
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

# Setup the robust optimization model solver=GurobiSolver(OutputFlag=0)
m = RobustModel()

# Variables
@defVar(m, obj)  # Put objective as constraint using dummy variable
@defVar(m, x[1:NUM_ASSET] >= 0)
    
  # Uncertainties
@defUnc(m,       r[1:NUM_ASSET]      )  # The returns
@defUnc(m, -1 <= z[1:NUM_ASSET] <= 1 )  # The "standard normals"
@defUnc(m,  0 <= y[1:NUM_ASSET] <= 1 )  # |z|/box

@setObjective(m, Max, obj)

# Portfolio constraint
@addConstraint(m, sum([ x[i] for i=1:NUM_ASSET ]) == 1)

# The objective constraint - uncertain
@addConstraint(m, sum([ r[i]*x[i] for i=1:NUM_ASSET ]) - obj >= 0)
    
solve(m, prefer_cuts=true)

return getValue(x)

end  # end of function

NUM_ASSET= 1000
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

#eval_gamma(0)  # Nominal - no uncertainty
eval_gamma(3)  # Some protection
println("Obj Value:\t", getValue(future_z))