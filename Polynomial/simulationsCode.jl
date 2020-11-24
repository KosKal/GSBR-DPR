cd("INSERT_PATH")

using Distributions
using PolynomialRoots

include("ToolBox.jl")
include("predictionGSBR.jl")

# define map parameters
theta = [0.05, 2.55, 0., -0.99]
x0 = 1.
dataSeed = 2
n = 114

# noise mixture: f_{2,l}
w1 = 0.6
lam1 = 1e05
lam2 = 1e-02 * lam1

x = genData(n,theta,x0,w1,lam1,lam2,dataSeed)
# Simulation parameters, T: prediction horizon [T=0 --> Plain GSBReconstruction]
degree,maxiter,burnin,seed,T = 5,500000,250000,3,0

@time thetas, zp, sx0 = predictionGSBR(x,degree,maxiter,burnin,seed,T);


