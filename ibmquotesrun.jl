# 2 min quote data from IBM 1 day
# time series optimization
quotes = readdlm("ibmquotes2min.txt")

using JuMP, JuMPeR, DDUS, Distributions

using(RARIMA)
using RCall

oracle = UCSOracle(quotes,.1, .2)

# residual forecast from ARIMA 



#build model

m = RobustModel()
setDefaultOracle!(m, oracle)