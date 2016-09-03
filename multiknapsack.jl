using JuMP, JuMPeR, DDUS, Distributions

# generate multiple networks data

#generates some silly market data for example
function genReturnData(numAssets, numObs, range)
	#Single factor CAPM model
    z = range*randn(numObs) + .2
    betas = linspace(0, range, numAssets)
    z * betas' + .05*randn(1, numObs,numAssets)
end


function genReturnData2(numAssets, numObs, range)
	#Single factor CAPM model
    #z = range*randn(numObs) + .2
    #betas = linspace(0, range, numAssets)
    range*randn(numObs, numAssets)
end

# multi discrete knapsack optimization problem

#read in the adjacency matrix ?

# calculate the weighted network Laplacian matrices 

cost = 10

wts = randn(cost)
#costs = randn(cost,cost)
numAssets =15
numObs = 20
range= 25

costs =  genReturnData2(numAssets, numObs, range)

# generate the adjacency matrices

dense = 0.4;
density = int64(dense*numAssets*numAssets)

function genAdjmat(nAssets, density)
    a = zeros(nAssets, nAssets)
    # random index vector 
    for i=1:density
        ind = int64((nAssets-1)*rand())+1
        ind2 = int64((nAssets-1)*rand())+1
        a[ind, ind2] = 1
    end
    return a
end

# degree of each vertex is sum of the 1s in the row

function getLapmat(a)
    m= size(a)
    Deg=-a
    for i=1:m
        # row sum
        s = sum(a[i,:]) 
        Deg[i,i]=s
    end
    return Deg
end

Adj = genAdjmat(nAssets, density)


wts = getLapmat(Adj)

#profit compatibility 
d = zeros(cost,cost,cost,cost);


for i=1:numAssets
    for j=1:numAssets
        for k1=1:numObs
            for k2= 1:numObs
                c1 = costs[k1,i] 
                c2 = costs[k2,j]
                # profit
                d[i,k1, j,k2] = c1+c2
            end
            
        end
    end
end




# uncertainty set on the network Laplacian matrix 

oracle = ChiSqSet(wts, 0, 10, .1, .2)

#build model
m = RobustModel()
setDefaultOracle!(m, oracle)

# for loop for multiple scenarios

gamma=1


# upper bd of the constraint
capacity=400


# bin var
@defVar(m, xs[1:numAssets,1:numAssets], Bin)
@defVar(m, t)

#@defVar(m, sumt <= sum{t[i], i=1:numAssets})

for i=1:numAssets
    for j=1:numAssets
        for k1=1:numObs
            for k2= 1:numObs
                @addConstraint(m, costs[i,j]*xs[i,j] + d[i,k1, j,k2]*xs[i, k1]*xs[k2, j] >= t)
            end
        end
    end
end


@setObjective(m, Max, t)
for j=1:numAssets
    @addConstraint(m, sum{wts[i,j]*xs[i,j], i=1:numAssets} <= capacity)
end

println(solve(m, prefer_cuts=true))
println("Obj Value:\t", getObjectiveValue(m))
println("Portfolio:\t", getValue(xs))
println("Assignment matrix:\t", getValue(wts))
