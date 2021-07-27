library(VineCopula)
library(doRNG)
library(dplyr)

"""
	Generate synthetic data from a t-copula and take the maximum
"""

prefix = "/home/aboulin/Documents/stage/var_FMado/bivariate/output/"

target <- list()

fmado_  <- function(xvec, yvec, lambda){
    F_X = ecdf(xvec)(xvec)
    G_Y = ecdf(yvec)(yvec)

    value_ = 0.5 * mean(abs(F_X^lambda - G_Y^(1-lambda)))
    return(value_)
}

target$generate_randomness <- function(nobservation){
    sample_ = BiCopSim(nobservation * nmax, 2, 0.8, 3)
    return(sample_)
}

target$robservation <- function(randomness){
    sample = apply(matrix(t(randomness), ncol = nmax),1, max)
    return(matrix(sample, ncol = 2, byrow = T))
}

M = 100 # number of iteration
n = c(256) # length of sample
nmax_ = c(16,32,64,128,256,512)

filename <- paste0(prefix, "max_student_M", M, "_n", nmax_[6], ".txt")

simu = function(target){
	foreach(rep = 1:M, .combine = rbind) %dorng% {
		# foreach is a function that create a loop and we, at each iteration, we increment the matrix of results (here output)
		# using rbind.
		# Allocate space for output

		FMado_store = matrix(0, length(n))
		FMado_runtimes = rep(0, length(n))
		FMado_lambda = rep(0, length(n))

		# generate all observations and sets of randomness to be used
		obs_rand = target$generate_randomness(max(n)) # we produce our bivariate vector of data
		obs_all = target$robservation(obs_rand) # some data are now hidden

		for(i in 1:length(n)){
			t_FMado = proc.time() # we compute the time to estimate
			# subset observations
			obs = obs_all[1:n[i],] # we pick the n[i] first rows, i.e 50 rows for the first, 100 for the second
			### We compute the lambda FMadogram
			FMado = fmado_(obs[,1], obs[,2], lambda) # we compute now the lambda-FMadogram (the normalized one)
			t_FMado = proc.time() - t_FMado

			# Save the results
			FMado_store[i,] = FMado
			FMado_runtimes[i] = t_FMado[3]
			FMado_lambda[i] = lambda
        }

		output = cbind(FMado_store, FMado_runtimes,FMado_lambda, n,nmax)
		output
	}
}

lambda_ = seq(0.01, 0.99, length.out = 100)
store_ = matrix(ncol = 6, nrow = 0)
for (i in 1:length(nmax_)){
	nmax = nmax_[i]
	print(i)
	lambda_FMadogram = foreach(rep = 1:length(lambda_), .combine = rbind) %dorng% {
			lambda = lambda_[rep]
			prod = simu(target)
			scaled = (prod[,1]) * sqrt(prod[,4])
			output = cbind(prod, scaled)
	}
	store_ = rbind(store_,lambda_FMadogram)
	
}
print(store_)
df_FMado = data.frame(store_)
names(df_FMado) = c("FMado", "runtime", "lambda", "n", "nmax", "scaled")
var = df_FMado %>% group_by(lambda, nmax) %>% summarise(var_emp = var(scaled))
print(var)
require('reticulate')

py_save_object(var, filename)

