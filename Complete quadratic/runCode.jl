cd("/home/kkaloudis/Documents/Mathematics/PhD/Julia/Prediction/fq pred")

using Distributions
using Polynomials
using DataFrames

include("toolBox.jl")
include("fqPredictionGSB.jl") # GSB
include("fqPredictionPar.jl") # Parametric

#########################################################
# Parameter Declaration
#########################################################

# map parameters
theta, x0, y0 = [1.38, 0., 0.27, 0., -1., 0.], -0.5, 1.2
# theta, x0, y0 = [1, 0., 0.3, 0., -1.4, 0.], 0.7, 0.7
# theta, x0, y0 = [1, 0., 0.23, 0., -1.31, 0.], 0.7, 0.5
# theta, x0, y0 = [1., 0., 0., 0., -1.8, 0.], 0.7, 0.5

# sample size
ss = 200
# prediction horizon 
T = 25

# noise f_{2,l} precisions
w1, lam1 = 0.7, 1e07
lam2 = 1e02

# corresponding deterministic orbit
xdet = zeros(ss+T)
xdet[1] = henon(theta, x0, y0)
xdet[2] = henon(theta, xdet[1], x0)
for i in 3:(ss+T)
  xdet[i] = henon(theta, xdet[i-1], xdet[i-2])
end

# dataSeed = 122 #f_{1}
# dataSeed = 46512 #f_{2,1}
dataSeed = 202011 #f_{2,2}
# dataSeed = 4517 #f_{2,3}
# dataSeed = 20205 #f_{2,4}

# fulldata = copy(xdet) # no noise
# fulldata = genDataf1(ss + T, theta, x0, y0, lam2, dataSeed) #f_{1} noise
fulldata = genData(ss + T, theta, x0, y0, w1, lam1, lam2, dataSeed) #f_{2,l} noise

predValues = copy(fulldata[(ss+1):(ss+T)])
data = copy(fulldata[1:ss])


# GSBR reconstruction - Prediction
gibbsIter, burnIn, gEps1, gEps2, thLow, thUp, zLow, zUp, papr, pbpr, thin, samplerSeed, filename =
 500000, 100000, 1e-03, 1e-03, -10.0, 10.0, -2.0, 2.0, 0.5, 0.5, 5, 12345, "/Results";

savelocation = string(pwd(), filename, "/GSB/seed$samplerSeed")
mkpath(savelocation)
writedlm(string(savelocation,"/predValues.txt"), predValues)
writedlm(string(savelocation,"/data.txt"), data, '\n')
@time thetas = fqPredictionGSB(data, T, gibbsIter, burnIn, gEps1, gEps2, thLow, thUp, zLow, zUp, papr, pbpr, thin, samplerSeed);
thetas = 0
gc()

# Parametric reconstruction - Prediction
fulldata = genData(ss + T, theta, x0, y0, w1, lam1, lam2, dataSeed) #f_{2,l} noise
predValues = copy(fulldata[(ss+1):(ss+T)])
data = copy(fulldata[1:ss])
savelocation = string(pwd(), filename, "/Par/seed$samplerSeed")
mkpath(savelocation)
writedlm(string(savelocation,"/predValues.txt"), predValues)
writedlm(string(savelocation,"/data.txt"), data, '\n')
@time thetas = fqPredictionPar(data, T, gibbsIter, burnIn, gEps1, gEps2, thLow, thUp, zLow, zUp, thin, samplerSeed);
thetas = 0
gc()



