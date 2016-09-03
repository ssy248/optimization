using JuMP, JuMPeR, DDUS, Distributions

#generates some random weights between lv and lv+rng
function genWeightData(numc, numw)
	#Single factor CAPM model
    randn(numw)
end

cost = 10

wts = randn(cost)
costs = randn(cost)

##########
srand(8675309)

wt_data = genWeightData(cost, cost)


# find the support of the distribution alphas


# upper bd of the constraint
capacity=400


#oracle = FBOracle(wt_data,.1, .2, .2)

sigma1= 0.2*rand(cost) .*wts
gamma=4

oracle = ChiSqSet(wts, .1)

#build model
m = RobustModel()
setDefaultOracle!(m, oracle)



# bin var
@defVar(m, xs[1:cost], Bin)
@defVar(m, 0<=t <=500)
@defUnc(m, 0<= z[1:cost] <= 1)
@addConstraint(m, sum{wts[i] * xs[i], i=1:cost} <= capacity)
@addConstraint(m, sum{costs[i] * xs[i], i=1:cost} >= t)
@setObjective(m, Max, t)
# polyhedral uncertainty
#setDefaultOracle!(m, oracle)
@addConstraint(m, sum{z[i], i=1:cost} <= gamma)

println(solve(m, prefer_cuts=true))
println("Obj Value:\t", getObjectiveValue(m))
println("Portfolio:\t", getValue(xs))