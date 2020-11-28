# Author: Duzhe Wang
# Last updated date:  10/11/2020
# Scenario 5: nonlinear decision rule    

rm(list=ls())
set.seed(2019)
funcdir="/Users/chang_file/Desktop/itrsimu/func/"
savedir<-"/Users/chang_file/Desktop/itrsimu/s5p100/" 
source(paste0(funcdir, "indirbst.R", sep="")) 
source(paste0(funcdir, "dlearningbst.R", sep="")) 
source(paste0(funcdir, "owlbst.R", sep="")) 
source(paste0(funcdir, "qlearning.R", sep="")) 
source(paste0(funcdir, "l1pls.R", sep="")) 
source(paste0(funcdir, "dlearning.R", sep="")) 
source(paste0(funcdir, "owl.R", sep="")) 


##---------Set parameters------------------##
K=10 # number of folds in cross validation
main="linear" # used in boosting algorithm 3
p=100  # covariate dimension
ntest<-3000  # testing data sample size
ntrvec=c(400, 800, 1200)  # training data sample size
nSim<-100  # simulation times
etavec=c(0.1, 0.15, 0.2, 0.25, 0.3)
roundsvec=seq(from=5, to=100, by=5)
treedepthvec=c(3,4,5,6)
lambda_seq=10^seq(2, -2, by = -.1)


##-----------Data generating process--------##
datagen<-function(n, p){
  # generate X from uniform distribution
  X<-matrix(runif(n* p, min=-1, max=1), nrow = n) 
  
  # generate A with equal probability (0.5) and use labels -1 and +1. 
  A<-rbinom(n, 1, 0.5)*2-1
  
  # generate Y from normal distribution
  pdg<-1-X[,1]*X[,1]*X[,1]+exp(X[,3]*X[,3]+X[,5])+0.6*X[,6]-(X[,7]+X[,8])*(X[,7]+X[,8])
  y<- 1+2*X[,1]+X[,2]+0.5*X[,3]+pdg*A+rnorm(n)
  
  data<-list(X=X,  A=A, y=y)
  return(data)
}

##-----------True optimal ITR----------------##

trueoptITR<-function(X){
  pdg<-1-X[,1]*X[,1]*X[,1]+exp(X[,3]*X[,3]+X[,5])+0.6*X[,6]-(X[,7]+X[,8])*(X[,7]+X[,8])
  trueoptITR<-(pdg>0)*2-1   
  return(trueoptITR)
}


#############################################################
##                                                         ##
##           DON'T CHANGE THE FOLLOWING CODE               ##
##                                                         ##
#############################################################


##-------Start of parameter tuning--------------------------##
## tune eta, rounds and tree depth
indirbst_tuneparams=matrix(NA, nrow = length(ntrvec), ncol=3)
dlearningbst_tuneparams=matrix(NA, nrow=length(ntrvec), ncol=3)
owlbst_tuneparams=matrix(NA, nrow=length(ntrvec), ncol=3)
## tune lambda
l1pls_tuneparams=rep(NA, length(ntrvec))
regudlearning_tuneparams=rep(NA, length(ntrvec))

for (i in 1:length(ntrvec)){
  ## generate data
  ntr=ntrvec[i]
  traindata=datagen(n=ntr, p=p)
  Xtrain<-traindata$X
  Atrain<-traindata$A
  ytrain<-traindata$y
  
  ## indirbst: algorithm 1
  print(paste("(indirbst)Current tuning sample size:", ntr))
  indirbst_result=indirbst_cvtune(ytrain, Atrain, Xtrain, K, etavec, roundsvec, treedepthvec)
  indirbst_tuneparams[i, 1]=indirbst_result$indirbst_opteta
  indirbst_tuneparams[i, 2]=indirbst_result$indirbst_optrounds
  indirbst_tuneparams[i, 3]=indirbst_result$indirbst_optdepth
  
  ## dlearningbst: algorithm 2
  print(paste("(dlearningbst)Current tuning sample size:", ntr))
  dlearningbst_result=dlearningbst_cvtune(ytrain, Atrain, Xtrain, K, etavec, roundsvec, treedepthvec)
  dlearningbst_tuneparams[i, 1]=dlearningbst_result$dlearningbst_opteta
  dlearningbst_tuneparams[i, 2]=dlearningbst_result$dlearningbst_optrounds
  dlearningbst_tuneparams[i, 3]=dlearningbst_result$dlearningbst_optdepth
  
  ## owlbst: algorithm 3
  print(paste("(owlbst)Current tuning sample size:", ntr))
  owlbst_result=owlbst_cvtune(ytrain, Atrain, Xtrain, K, etavec, roundsvec, treedepthvec, main)
  owlbst_tuneparams[i, 1]=owlbst_result$owlbst_opteta
  owlbst_tuneparams[i, 2]=owlbst_result$owlbst_optrounds
  owlbst_tuneparams[i, 3]=owlbst_result$owlbst_optdepth
  
  ## l1pls
  print(paste("(l1pls)Current tuning sample size:", ntr))
  l1pls_result=l1pls_cvtune(ytrain, Atrain, Xtrain, K, lambda_seq)
  l1pls_tuneparams[i]=l1pls_result$l1pls_optlambda
  
  ## dlearning when p is large
  print(paste("(dlearning)Current tuning sample size:", ntr))
  regudlearning_reuslt=regudlearning_cvtune(ytrain, Atrain, Xtrain, K, lambda_seq)
  regudlearning_tuneparams[i]=regudlearning_reuslt$regudlearning_optlambda
}


##------Start of simulation-----------------------------------##

## Step 1: make matrices to save results
qlearning_errMat<-matrix(NA, nrow=length(ntrvec), ncol=nSim)
qlearning_valueMat<-matrix(NA, nrow=length(ntrvec), ncol=nSim)

l1pls_errMat<-matrix(NA, nrow=length(ntrvec), ncol=nSim)
l1pls_valueMat<-matrix(NA, nrow=length(ntrvec), ncol=nSim)

dlearning_errMat<-matrix(NA, nrow=length(ntrvec), ncol=nSim)
dlearning_valueMat<-matrix(NA, nrow=length(ntrvec), ncol=nSim)

owl_linear_errMat<-matrix(NA, nrow=length(ntrvec), ncol=nSim)
owl_linear_valueMat<-matrix(NA, nrow=length(ntrvec), ncol=nSim)

owl_rbf_errMat<-matrix(NA, nrow=length(ntrvec), ncol=nSim)
owl_rbf_valueMat<-matrix(NA, nrow=length(ntrvec), ncol=nSim)

indirbst_errMat<-matrix(NA, nrow=length(ntrvec), ncol=nSim)
indirbst_valueMat<-matrix(NA, nrow=length(ntrvec), ncol=nSim)

dlearningbst_errMat<-matrix(NA, nrow=length(ntrvec), ncol=nSim)
dlearningbst_valueMat<-matrix(NA, nrow=length(ntrvec), ncol=nSim)

owlbst_errMat<-matrix(NA, nrow=length(ntrvec), ncol=nSim)
owlbst_valueMat<-matrix(NA, nrow=length(ntrvec), ncol=nSim)

## Step 2: testing data
testdata=datagen(n=ntest, p=p)
Xtest<-testdata$X
Atest<-testdata$A
ytest<-testdata$y
## treatment assignment from true optimal ITR
trueoptITR.test<-trueoptITR(X=Xtest) 
# evaluate the value function of true optimal ITR
indi<-as.integer(trueoptITR.test==Atest)
value.numerator<-mean(2*ytest*indi)
value.denominator<-mean(2*indi)
EstimatedValueofTrueITR<-value.numerator/value.denominator

## Step 3: run simulation 
for (nn in 1:length(ntrvec)){
  ntr=ntrvec[nn]
  
  for (iter in 1:nSim) {
    ## generate training data 
    traindata=datagen(n=ntr, p=p)
    Xtrain<-traindata$X
    Atrain<-traindata$A
    ytrain<-traindata$y
    
    ## qlearning
    print(paste("(qlearning)Current sample size:", ntr, "Current iteration:", iter))
    qlearning_result=qlearning_train_test(ytrain, Atrain, Xtrain,trueoptITR.test, Atest, Xtest, ytest)
    qlearning_errMat[nn, iter]=qlearning_result$err
    qlearning_valueMat[nn, iter]=qlearning_result$estvalue
    
    ## l1pls
    print(paste("(l1pls)Current sample size:", ntr, "Current iteration:", iter))
    lambda=l1pls_tuneparams[nn]
    l1pls_result=l1pls_train_test(ytrain, Atrain, Xtrain, trueoptITR.test, Atest, Xtest, ytest, lambda)
    l1pls_errMat[nn, iter]=l1pls_result$err
    l1pls_valueMat[nn, iter]=l1pls_result$estvalue
    
    ## dlearning
    print(paste("(dlearning)Current sample size:", ntr, "Current iteration:", iter))
    lambda=regudlearning_tuneparams[nn]
    dlearning_result=regudlearning_train_test(ytrain, Atrain, Xtrain, trueoptITR.test, Atest, Xtest, ytest, lambda)
    dlearning_errMat[nn, iter]=dlearning_result$err
    dlearning_valueMat[nn, iter]=dlearning_result$estvalue
    
    ## linear owl
    print(paste("(linear owl)Current sample size:", ntr, "Current iteration:", iter))
    owl_linear_result=OWLlinear_train_test(ytrain, Atrain, Xtrain, trueoptITR.test, Atest, Xtest, ytest)
    owl_linear_errMat[nn, iter]=owl_linear_result$err
    owl_linear_valueMat[nn, iter]=owl_linear_result$estvalue
    
    ## rbf owl
    print(paste("(rbf owl)Current sample size:", ntr, "Current iteration:", iter))
    owl_rbf_result=OWLRBF_train_test(ytrain, Atrain, Xtrain, trueoptITR.test, Atest, Xtest, ytest)
    owl_rbf_errMat[nn, iter]=owl_rbf_result$err
    owl_rbf_valueMat[nn, iter]=owl_rbf_result$estvalue
    
    ## indirbst
    print(paste("(indirbst)Current sample size:", ntr, "Current iteration:", iter))
    eta=indirbst_tuneparams[nn, 1]
    rounds=indirbst_tuneparams[nn, 2]
    treedepth=indirbst_tuneparams[nn, 3]  
    indirbst_result=indirbst_train_test(ytrain, Atrain, Xtrain, trueoptITR.test, Atest, Xtest, ytest, eta, rounds, treedepth)
    indirbst_errMat[nn, iter]=indirbst_result$err
    indirbst_valueMat[nn, iter]=indirbst_result$estvalue
    
    ## dlearningbst
    print(paste("(dlearningbst)Current sample size:", ntr, "Current iteration:", iter))
    eta=dlearningbst_tuneparams[nn, 1]
    rounds=dlearningbst_tuneparams[nn, 2]
    treedepth=dlearningbst_tuneparams[nn, 3]  
    dlearningbst_result=dlearningbst_train_test(ytrain, Atrain, Xtrain, trueoptITR.test, Atest, Xtest, ytest, eta, rounds, treedepth)
    dlearningbst_errMat[nn, iter]=dlearningbst_result$err
    dlearningbst_valueMat[nn, iter]=dlearningbst_result$estvalue
    
    ## owlbst
    print(paste("(owlbst)Current sample size:", ntr, "Current iteration:", iter))
    eta=owlbst_tuneparams[nn, 1]
    rounds=owlbst_tuneparams[nn, 2]
    treedepth=owlbst_tuneparams[nn, 3]  
    owlbst_result=owlbst_train_test(ytrain, Atrain, Xtrain,trueoptITR.test, Atest, Xtest, ytest, eta, rounds, treedepth, main)
    owlbst_errMat[nn, iter]=owlbst_result$err
    owlbst_valueMat[nn, iter]=owlbst_result$estvalue
  }
}


##------Analysis of results-----------------------------##

## misclassification error rate and value 

qlearning_error=transform(qlearning_errMat, mean=apply(qlearning_errMat, 1, mean, na.rm=TRUE), 
                          sd=apply(qlearning_errMat, 1, sd, na.rm=TRUE))
qlearning_value=transform(qlearning_valueMat, mean=apply(qlearning_valueMat, 1, mean, na.rm=TRUE), 
                          sd=apply(qlearning_valueMat, 1, sd, na.rm=TRUE))


l1pls_error=transform(l1pls_errMat, mean=apply(l1pls_errMat, 1, mean, na.rm=TRUE), 
                      sd=apply(l1pls_errMat, 1, sd, na.rm=TRUE))
l1pls_value=transform(l1pls_valueMat, mean=apply(l1pls_valueMat, 1, mean, na.rm=TRUE), 
                      sd=apply(l1pls_valueMat, 1, sd, na.rm=TRUE))


dlearning_error=transform(dlearning_errMat, mean=apply(dlearning_errMat, 1, mean, na.rm=TRUE), 
                          sd=apply(dlearning_errMat, 1, sd, na.rm=TRUE))
dlearning_value=transform(dlearning_valueMat, mean=apply(dlearning_valueMat, 1, mean, na.rm=TRUE), 
                          sd=apply(dlearning_valueMat, 1, sd, na.rm=TRUE))


owl_linear_error=transform(owl_linear_errMat, mean=apply(owl_linear_errMat, 1, mean, na.rm=TRUE), 
                           sd=apply(owl_linear_errMat, 1, sd, na.rm=TRUE))
owl_linear_value=transform(owl_linear_valueMat, mean=apply(owl_linear_valueMat, 1, mean, na.rm=TRUE), 
                           sd=apply(owl_linear_valueMat, 1, sd, na.rm=TRUE))


owl_rbf_error=transform(owl_rbf_errMat, mean=apply(owl_rbf_errMat, 1, mean, na.rm=TRUE), 
                        sd=apply(owl_rbf_errMat, 1, sd, na.rm=TRUE))
owl_rbf_value=transform(owl_rbf_valueMat, mean=apply(owl_rbf_valueMat, 1, mean, na.rm=TRUE), 
                        sd=apply(owl_rbf_valueMat, 1, sd, na.rm=TRUE))


indirbst_error=transform(indirbst_errMat, mean=apply(indirbst_errMat, 1, mean, na.rm=TRUE), 
                         sd=apply(indirbst_errMat, 1, sd, na.rm=TRUE))
indirbst_value=transform(indirbst_valueMat, mean=apply(indirbst_valueMat, 1, mean, na.rm=TRUE), 
                         sd=apply(indirbst_valueMat, 1, sd, na.rm=TRUE))


dlearningbst_error=transform(dlearningbst_errMat, mean=apply(dlearningbst_errMat, 1, mean, na.rm=TRUE), 
                             sd=apply(dlearningbst_errMat, 1, sd, na.rm=TRUE))
dlearningbst_value=transform(dlearningbst_valueMat, mean=apply(dlearningbst_valueMat, 1, mean, na.rm=TRUE), 
                             sd=apply(dlearningbst_valueMat, 1, sd, na.rm=TRUE))

owlbst_error=transform(owlbst_errMat, mean=apply(owlbst_errMat, 1, mean, na.rm=TRUE), 
                       sd=apply(owlbst_errMat, 1, sd, na.rm=TRUE))
owlbst_value=transform(owlbst_valueMat, mean=apply(owlbst_valueMat, 1, mean, na.rm=TRUE), 
                       sd=apply(owlbst_valueMat, 1, sd, na.rm=TRUE))

##----------------------SAVE OUTPUT--------------------------------##

## save tuning results 
saveRDS(indirbst_tuneparams, file=paste(savedir,"indirbst_tuneparams.rds", sep=""))
saveRDS(dlearningbst_tuneparams, file=paste(savedir,"dlearningbst_tuneparams.rds", sep=""))
saveRDS(owlbst_tuneparams, file=paste(savedir,"owlbst_tuneparams.rds", sep=""))
saveRDS(l1pls_tuneparams, file=paste(savedir,"l1pls_tuneparams.rds", sep=""))
saveRDS(regudlearning_tuneparams, file=paste(savedir,"regudlearning_tuneparams.rds", sep=""))

## save training results 
saveRDS(qlearning_error, file=paste(savedir,"qlearning_error.rds", sep=""))
saveRDS(qlearning_value, file=paste(savedir,"qlearning_value.rds", sep=""))
saveRDS(l1pls_error, file=paste(savedir,"l1pls_error.rds", sep=""))
saveRDS(l1pls_value, file=paste(savedir,"l1pls_value.rds", sep=""))
saveRDS(dlearning_error, file=paste(savedir,"dlearning_error.rds", sep=""))
saveRDS(dlearning_value, file=paste(savedir,"dlearning_value.rds", sep=""))
saveRDS(owl_linear_error, file=paste(savedir,"owl_linear_error.rds", sep=""))
saveRDS(owl_linear_value, file=paste(savedir,"owl_linear_value.rds", sep=""))
saveRDS(owl_rbf_error, file=paste(savedir,"owl_rbf_error.rds", sep=""))
saveRDS(owl_rbf_value, file=paste(savedir,"owl_rbf_value.rds", sep=""))
saveRDS(indirbst_error, file=paste(savedir,"indirbst_error.rds", sep=""))
saveRDS(indirbst_value, file=paste(savedir,"indirbst_value.rds", sep=""))
saveRDS(dlearningbst_error, file=paste(savedir,"dlearningbst_error.rds", sep=""))
saveRDS(dlearningbst_value, file=paste(savedir,"dlearningbst_value.rds", sep=""))
saveRDS(owlbst_error, file=paste(savedir,"owlbst_error.rds", sep=""))
saveRDS(owlbst_value, file=paste(savedir,"owlbst_value.rds", sep=""))

## save estimated value of the true optimal ITR based on the testing data
saveRDS(EstimatedValueofTrueITR, file=paste(savedir,"EstimatedValueofTrueITR.rds", sep=""))

