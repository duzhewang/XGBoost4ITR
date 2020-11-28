## Author: Duzhe Wang
## Last updated date: 10/10/2020
## Outcome weighted learning: hinge loss (Zhao et al 2012), use DynTxRegime package 
## p(A|x)=0.5
## Note we don't tune paramaters in OWL. 


library(DynTxRegime)

myPredict<- function(object, newx){
  n<-nrow(newx)
  return(rep(0.5, n))   # make P(A=1|X)=P(A=-1|X)=0.5
} 


#################################################
##                   linear OWL                ##
#################################################

OWLlinear_train_test=function(ytrain, Atrain, Xtrain, trueoptITR.test, Atest, Xtest, ytest){
  reward<-ytrain-min(ytrain) # make reward positive
  newdata<-data.frame(Xtrain, A=as.factor(Atrain))
  moPropen <- buildModelObj(model =as.formula(paste("~",paste("X",1:ncol(Xtrain),sep="",collapse="+"))) ,
                            solver.method = 'glm',
                            solver.args = list('family'='binomial'),
                            predict.method = 'myPredict', 
                            predict.args = list('newx'='newdata'))
  ## training 
  ## use the default lambda=2
  fitOWL <- owl(moPropen = moPropen, data = newdata, reward = reward, txName = 'A',
                regime = as.formula(paste("~",paste("X",1:ncol(Xtrain),sep="",collapse="+"))), 
                surrogate = 'hinge', kernel = 'linear', kparam = NULL, sigf=5, verbose=0)
  
  # prediction
  #decisioncoef=regimeCoef(fitOWL)
  #estf<-Xtest%*%decisioncoef[2:(p+1)]+decisioncoef[1]
  #pred_trt=rep(1, ntest)
  #pred_trt[estf<0]=-1
  
  pred<-optTx(fitOWL, data.frame(Xtest))
  pred_trt<-pred$optimalTx

  
  # calculate the misclassification rate: 
  # compare the true optimal treatment assignment with the estimated optimal treatment assignment
  err<- mean(trueoptITR.test!=pred_trt)
  
  # evaluate the value function of estimated optimal ITR
  indi<-as.integer(pred_trt==Atest)
  estvalue.numerator<-mean(2*ytest*indi)
  estvalue.denominator<-mean(2*indi)
  estvalue<-estvalue.numerator/estvalue.denominator
  
  ##--------RETURN OUTPUT------------##
  ## return value par in fitOWL is the estimated coefficients for the owl model 
  return(list(fitOWL=fitOWL, pred_trt=pred_trt, err=err, estvalue=estvalue))
}



#####################################################
##               nonlinear OWL                     ##
#####################################################

OWLRBF_train_test=function(ytrain, Atrain, Xtrain, trueoptITR.test, Atest, Xtest, ytest){
  
  reward<-ytrain-min(ytrain) # make reward positive
  newdata<-data.frame(Xtrain, A=as.factor(Atrain))
  
  moPropen <- buildModelObj(model =as.formula(paste("~",paste("X",1:ncol(Xtrain), sep="",collapse="+"))) ,
                            solver.method = 'glm',
                            solver.args = list('family'='binomial'),
                            predict.method = 'myPredict', 
                            predict.args = list('newx'='newdata'))
  
  # kparam is the inverse bandwidth(1/sigma)
  discvec=as.vector(as.matrix(dist(Xtrain)))
  lowerindex=which(lower.tri(as.matrix(dist(Xtrain)), diag=FALSE)==TRUE)
  fitOWL <- owl(moPropen = moPropen,
                data = newdata, reward = reward, txName = 'A',
                regime = as.formula(paste("~",paste("X",1:ncol(Xtrain),sep="",collapse="+"))),
                surrogate = 'hinge', kernel = 'radial', kparam = 1/median(discvec[lowerindex]), sigf=5, verbose=0)
  
  
  # prediction
  pred<-optTx(fitOWL, data.frame(Xtest))
  pred_trt<-pred$optimalTx
  
  
  # calculate the misclassification rate: 
  # compare the true optimal treatment assignment with the estimated optimal treatment assignment
  err<- mean(trueoptITR.test!=pred_trt)
  
  # evaluate the value function of estimated optimal ITR
  indi<-as.integer(pred_trt==Atest)
  estvalue.numerator<-mean(2*ytest*indi)
  estvalue.denominator<-mean(2*indi)
  estvalue<-estvalue.numerator/estvalue.denominator
  
  ##--------RETURN OUTPUT------------##
  return(list(fitOWL=fitOWL, pred_trt=pred_trt, err=err, estvalue=estvalue))
}




















