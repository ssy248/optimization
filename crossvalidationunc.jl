# cross validation for uncertainty sets

# Here, we consider a simple model: using a mean vector to represent
# a set of samples. The goodness of the model is assessed in terms
# of the RMSE (root-mean-square-error) evaluated on the testing set
#

using MLBase

# functions
compute_center(X::Matrix{Float64}) = vec(mean(X, 2))

compute_rmse(c::Vector{Float64}, X::Matrix{Float64}) =
    sqrt(mean(sum(abs2(X .- c),1)))

# number of training samples
ntr = 5
# number of test samples
nte = 2
d=108 # row dim of observations

# data
const n = 200
const data = [2., 3.] .+ randn(2, n)


Xtr = randn(ntr, d)
ytr= Xtr*

data= matrix; 

# let c be the value that is allocated to each asset 
c = solution;

# cross validation
scores = cross_validate(
    inds -> compute_center(data[:, inds]),        # training function
    (c, inds) -> compute_rmse(c, data[:, inds]),  # evaluation function
    n,              # total number of samples
    Kfold(n, 8),    # cross validation plan: 8-fold
    Reverse) 		# smaller score indicates better model



# get the mean and std of the scores
(m, s) = mean_and_std(scores)

# display results
@printf("best model = (%.4f, %.4f), score = %.6f\n", c[1], c[2], v)