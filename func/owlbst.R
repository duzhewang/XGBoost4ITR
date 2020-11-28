## Author: Duzhe Wang
## Last updated date: 10/10/2020
## Algorithm 3 in our paper


library(tidyr)
library(xgboost)

## Tuning 

owlbst_validate=function(trainingX, trainingA, trainingy, valiX, valiA, valiy, eta, rounds, treedepth, main){
  
  ## validation data matrix
  valiMat=data.frame(valiA, valiX)
  valiMat.xgb=list()
  valiMat.xgb$data=valiMat[, -1]
  valiMat.xgb$label=valiMat[, 1] 
  dvali=xgb.DMatrix(as.matrix(valiMat.xgb$data), label = as.vector(valiMat.xgb$label))
  
  ## training data matrix
  trainMat=data.frame(trainingA,trainingX)
  trainMat.xgb=list()
  trainMat.xgb$data=trainMat[,-1]
  trainMat.xgb$label=trainMat[, 1]
  dtrain=xgb.DMatrix(as.matrix(trainMat.xgb$data), label = as.vector(trainMat.xgb$label))
  
  ## do linear regression to calculate the residuals
  if (main=="linear"){
    ## linear regression model
    regdata<-data.frame(y=trainingy, x=trainingX)
    regmodel<-lm(y~., data=regdata)
    resi<-regmodel$residuals
  }
  if (main=="null"){
    ## null model 
    resi=trainingy-mean(trainingy)
  }
  
  # define deviance loss function: given prediction, return gradient and second order gradient
  obj=function(preds, dtrain){
    labels <- getinfo(dtrain, "label")
    weights<- 2*abs(resi)
    signs<-as.numeric(resi>0)*2-1
    grad<-weights*(-2)*labels*signs*exp(-2*labels*signs*preds)/(1+exp(-2*labels*signs*preds))
    hess<-4*weights*exp(-2*labels*signs*preds)/((1+exp(-2*labels*signs*preds))^2)
    return(list(grad = grad, hess = hess))
  }
  
  ## training 
  param <- list(max_depth=treedepth, eta=eta, nthread = 2, verbosity=0, objective=obj)
  num_round<-rounds
  bst <- xgb.train(param, dtrain, num_round)
  
  ## prediction
  pred_value<-predict(bst, dvali) # the output is f(x)
  pred_trt<-as.numeric(pred_value>0)*2-1
  
  # tuning criterion: evaluate the value function 
  indi<-as.integer(pred_trt==valiA)
  estvalue.numerator<-mean(2*valiy*indi)
  estvalue.denominator<-mean(2*indi)
  estvalue<-estvalue.numerator/estvalue.denominator
  return(estvalue)
}


owlbst_cvtune<-function(ytrain, Atrain, Xtrain, K, etavec, roundsvec, treedepthvec, main){
  ##----form a matrix-------------------##
  paramat<-crossing(etavec, roundsvec, treedepthvec)
  paramat<-as.matrix(paramat)
  colnames(paramat)<-NULL
  foldlen<-length(ytrain)/K   # length of each fold
  estvaluemat=matrix(NA, nrow=nrow(paramat), ncol=K)
  
  for(i in 1:nrow(paramat)){
    eta=paramat[i,1]
    rounds=paramat[i,2]
    treedepth=paramat[i, 3]
    for(k in 1:K){
      index=((k-1)*foldlen+1):(k*foldlen)
      trainingX=Xtrain[-index,]
      trainingA=Atrain[-index]
      trainingy=ytrain[-index]
      valiX=Xtrain[index,]
      valiA=Atrain[index]
      valiy=ytrain[index]
      estvaluemat[i, k]=owlbst_validate(trainingX, trainingA, trainingy, valiX, valiA, valiy, eta, rounds, treedepth, main)
    }
  }
  
  owlbst_estvaluemat=cbind(estvaluemat, "means"=rowMeans(estvaluemat, na.rm=TRUE))
  owlbst_opteta=paramat[which.max(owlbst_estvaluemat[, "means"]), 1]
  owlbst_optrounds=paramat[which.max(owlbst_estvaluemat[, "means"]), 2]
  owlbst_optdepth=paramat[which.max(owlbst_estvaluemat[, "means"]), 3]
  
  ##---------Output------------------## 
  return(list(owlbst_estvaluemat=owlbst_estvaluemat, 
              owlbst_opteta=owlbst_opteta, 
              owlbst_optrounds=owlbst_optrounds,
              owlbst_optdepth=owlbst_optdepth))
}



## Training 

owlbst_train_test=function(ytrain, Atrain, Xtrain,trueoptITR.test, Atest, Xtest, ytest, eta, rounds, treedepth, main){
  
  ## testing data matrix
  testMat=data.frame(Atest, Xtest)
  testMat.xgb=list()
  testMat.xgb$data=testMat[, -1]
  testMat.xgb$label=testMat[, 1] 
  dtest=xgb.DMatrix(as.matrix(testMat.xgb$data), label = as.vector(testMat.xgb$label))
  
  ## training data matrix
  trainMat=data.frame(Atrain,Xtrain)
  trainMat.xgb=list()
  trainMat.xgb$data=trainMat[,-1]
  trainMat.xgb$label=trainMat[, 1]
  dtrain=xgb.DMatrix(as.matrix(trainMat.xgb$data), label = as.vector(trainMat.xgb$label))
  
  ## do linear regression to calculate the residuals
  if (main=="linear"){
    regdata<-data.frame(y=ytrain, x=Xtrain)
    regmodel<-lm(y~., data=regdata)
    resi<-regmodel$residuals
  }
  if (main=="null"){
    resi=ytrain-mean(ytrain)
  }
  
  # define deviance loss function: given prediction, return gradient and second order gradient
  obj=function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    weights<- 2*abs(resi)
    signs<-as.numeric(resi>0)*2-1
    grad<-weights*(-2)*labels*signs*exp(-2*labels*signs*preds)/(1+exp(-2*labels*signs*preds))
    hess<-4*weights*exp(-2*labels*signs*preds)/((1+exp(-2*labels*signs*preds))^2)
    return(list(grad = grad, hess = hess))
  }
  
  ## training 
  param <- list(max_depth=treedepth, eta=eta, nthread = 2, verbosity=0, objective=obj)
  num_round<-rounds
  bst <- xgb.train(param, dtrain, num_round)
  
  ## prediction
  pred_value<-predict(bst, dtest) # the output is f(x)
  pred_trt<-as.numeric(pred_value>0)*2-1
  
  ## misclassification error rate
  ## compare the true optimal ITR with the estimated optimal ITR when applying to the testing data
  err <- mean(trueoptITR.test!=pred_trt)
  
  # evaluate the value function of estimated optimal ITR
  indi<-as.integer(pred_trt==Atest)
  estvalue.numerator<-mean(2*ytest*indi)
  estvalue.denominator<-mean(2*indi)
  estvalue<-estvalue.numerator/estvalue.denominator
  
  ##--------RETURN OUTPUT------------##
  return(list(bst=bst, pred_trt=pred_trt, err=err, estvalue=estvalue))
}








