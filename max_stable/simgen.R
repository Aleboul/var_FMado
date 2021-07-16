library(SpatialExtremes)

"""
    Simulate a Smith process and Estimate Lambda-FMadogram with the realisations.
"""

fmado_ <- function(X, lmbd){
    Nnb = ncol(X)
    Tnb = nrow(X)
    F_X = ecdf(X[,1])(X[,1])
    G_Y = ecdf(X[,2])(X[,2])
    lmado = 0.5*mean(abs(F_X^lmbd - G_Y^(1-lmbd))) - 0.5 * lmbd * mean(1-F_X^lmbd) - 0.5 * (1-lmbd) * mean(1-G_Y^(1-lmbd)) + 0.5 * (1-lmbd + lmbd * lmbd) / (2-lmbd) / (1+lmbd) 

    return(lmado)
}

simgen <- function(h, lmbd, niter, nsample, sigma){
    ans = list()
    x = rep(0, length(h))
    locations = cbind(x,h)
    colnames(locations) = c('lon', 'lat')
    horizon_ =  h[-1]
    for(n in 1:niter){
        print(n)
        lmbd_FMado_ = matrix(rep(0,length(horizon_) * length(lmbd)), nrow = length(horizon_), ncol = length(lmbd))
        sample_ = rmaxstab(nsample, locations, cov.mod = 'gauss', cov11 = sigma, cov12 = 0, cov22 = sigma)
        for(i in 1:length(horizon_)){
            for(j in 1:length(lmbd)){
                lmbd_FMado_[i,j] = fmado_(sample_[,c(1,i+1)], lmbd[j])
            }
        }
        ans[n] = list(lmbd_FMado_)
    }
    return(ans)
}

###### test ######

h = seq(0,20,length = 21)
lmbd = seq(0,1.0, length = 11)
sigma = 5
niter = 300
nsample = 1024
results = simgen(h, lmbd, niter, nsample, sigma)
print(results)
require('reticulate')

py_save_object(results, "/home/aboulin/Documents/stage/var_FMado/max_stable/data_1024.txt")
