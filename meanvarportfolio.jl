#####
# Times Series mean variance portfolio ex
#####
#Solves the mean variance portfolio allocation problem
##  min  -mu^T x + lambda x^T Q x
#   s.t. mu^T x >= t  for all  mu in U
#		max mu		
#        1^T x == 1
#        x >= 0
# where U is constructed from data
using JuMP, JuMPeR, DDUS, Distributions

using StatsBase

using DataFrames

# load data


#prices = int(open(readdlm,"petcodatafile.txt"))
#squeeze(prices,2)


ebb =readdlm("ebaycsv.csv",',')
# delete first row of label
ebb1 =ebb[1:size(ebb,1) .!= 1,: ]
# attributes

a1 =float64(ebb1[:, 2])
a2 = float64(ebb1[:, 3])
a3 =float64(ebb1[:, 4])

# expd returns , variance
ret= float64(zeros(108));

# calculate 
for i in 2:108
    # collection , value, index
    #setindex!(ret, prices[i]/(prices[i-1]-1), i)
    ret[i] = a1[i]/(a1[i-1]-1) 
    #ret[i] = int(ret[i])
end

r = hcat(ret, 10*a2, 10*a3)
#prices = float64(prices)

#rowsize = 108
#colsize = size(prices,2)





# cov
Q= cov(ret)

covar = cov(r)

# expected return
#expret = sum(ret)/rowsize


# construct portfolio m assets with wts: each stock unit divided by the total weight

#totsum = sum(ret)
#weights = ret/totsum



# total no of months
#timetot = rowsize


# oracle
oracle = LCXOracle(r, .1, .1)


# additional 
#sum1= x*covar[1,:]*x + x*covar[2,:]*x + x*covar[3,:]*x


# build model
c= covar[1,:] + covar[2,:] + covar[3,:]
sumc = c[1]+ c[2]+ c[3]
@defUnc(m, mu[1:3])
@defVar(m, 0<=t <=100)
@defVar(m, t1 >=0)
@defVar(m, x[1:3] >= 0)
@defVar(m, sum1[1:3] )
@defVar(m, 1 <= lambda <=10)

#@defVar(m, q[1:3] == x*covar[1,:]*x + x*covar[2,:]*x + x*covar[3,:]*x)
@addConstraint(m, sum(sum1)== sumc)
@addConstraint(m, sum(x) == 1)
@addConstraint(m, sum{-mu[i] * x[i], i=1:3}+ lambda*sumc <=t)


@setObjective(m, Min, t)

#prev code
#m = RobustModel()
#setDefaultOracle!(m, oracle)
# mean returns
#@defUnc(m, mu)
#@defVar(m, x >= 0)
#@defVar(m, 1 <= lambda <=10)
#@defVar(m, F == -mu'*x + lambda*x'*Q*x)

#@addConstraint(m, sum(x) == 1)

#@setObjective(m, Min, F)
#@setObjective(m, Max, mu)

#for now, must use cuts to solve
#Notice how set construction takes advantage of the correlations tructure
#So that allocation favors higher indices
println(solve(m, prefer_cuts=true))
println("Obj Value:\t", getObjectiveValue(m))
println("Portfolio:\t", getValue(xs))