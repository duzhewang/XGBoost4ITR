## Author: Duzhe Wang
## Last updated date:10/10/2020
## Algorithm 1 in our paper


library(tidyr)   # use crossing function
library(xgboost) # use xgboost

## Tuning 

indirbst_validate=function(trainingX, trainingA, trainingy, valiX, valiA, valiy, eta, rounds, treedepth){
  ## training data matrix
  trainMat=data.frame(trainingy, trainingA, trainingX)
  
  ## validation data matrix
  valiMat=data.frame(valiy, valiA, valiX)
  
  ## training 
  bayes.neg1<- xgboost(booster="gbtree", data = as.matrix(trainMat[trainMat$trainingA==-1,-c(1,2)]), 
                       label = trainMat$trainingy[trainMat$trainingA==-1], max_depth = treedepth, eta=eta, nthread = 2, 
                       nrounds=rounds, objective = "reg:squarederror", verbose = 0)  ## verbose=0 returns no message
  
  bayes.pos1<- xgboost(booster="gbtree", data = as.matrix(trainMat[trainMat$trainingA==1,-c(1,2)]), 
                       label = trainMat$trainingy[trainMat$trainingA==1], max_depth = treedepth, eta=eta, nthread = 2, 
                       nrounds=rounds, objective = "reg:squarederror", verbose = 0)
  
  ## prediction
  predMat<-matrix(data=NA, nrow=nrow(valiX), ncol=2)
  predMat[,1] <- predict(bayes.neg1, as.matrix(valiMat[-c(1,2)]))
  predMat[,2] <- predict(bayes.pos1, as.matrix(valiMat[-c(1,2)]))
  pred_trt <- apply(predMat, 1, which.max)
  pred_trt[pred_trt==1]=-1
  pred_trt[pred_trt==2]=1
  
  ## tuning criterion: evaluate the value function 
  indi<-as.integer(pred_trt==valiA)
  estvalue.numerator<-mean(2*valiy*indi)
  estvalue.denominator<-mean(2*indi)
  estvalue<-estvalue.numerator/estvalue.denominator
  return(estvalue)
}


indirbst_cvtune<-function(ytrain, Atrain, Xtrain, K, etavec, roundsvec, treedepthvec){
  # K: number of folds
  # etavec, roundsvec, treedepthvec: tuning parameters

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
      estvaluemat[i, k]=indirbst_validate(trainingX, trainingA, trainingy, valiX, valiA, valiy, eta, rounds, treedepth)
    }
  }
  
  indirbst_estvaluemat=cbind(estvaluemat, "means"=rowMeans(estvaluemat, na.rm=TRUE))
  indirbst_opteta=paramat[which.max(indirbst_estvaluemat[, "means"]), 1]
  indirbst_optrounds=paramat[which.max(indirbst_estvaluemat[, "means"]), 2]
  indirbst_optdepth=paramat[which.max(indirbst_estvaluemat[, "means"]), 3]
  
  ##---------Output------------------## 
  return(list(indirbst_estvaluemat=indirbst_estvaluemat, 
              indirbst_opteta=indirbst_opteta, 
              indirbst_optrounds=indirbst_optrounds,
              indirbst_optdepth=indirbst_optdepth))
}


## Training 
indirbst_train_test=function(ytrain, Atrain, Xtrain, trueoptITR.test, Atest, Xtest, ytest, eta, rounds, treedepth){
  
  ##-------training data-----------------##
  trainM <- data.frame(ytrain, Atrain, Xtrain)
  
  ##-------testing data----------------##
  testM=data.frame(ytest, Atest, Xtest)
  
  ## training 
  bayes.neg1<- xgboost(booster="gbtree", data = as.matrix(trainM[trainM$Atrain==-1,-c(1,2)]), 
                       label = trainM$ytrain[trainM$Atrain==-1], max_depth = treedepth, eta=eta, nthread = 2, 
                       nrounds=rounds, objective = "reg:squarederror", verbose = 0)
  
  bayes.pos1<- xgboost(booster="gbtree",data = as.matrix(trainM[trainM$Atrain==1,-c(1,2)]), 
                       label = trainM$ytrain[trainM$Atrain==1], max_depth = treedepth, eta=eta, nthread = 2, 
                       nrounds=rounds, objective = "reg:squarederror", verbose = 0)
  ## prediction
  predMat<-matrix(data=NA, nrow=nrow(Xtest), ncol=2)
  predMat[,1] <- predict(bayes.neg1, as.matrix(testM[-c(1,2)]))
  predMat[,2] <- predict(bayes.pos1, as.matrix(testM[-c(1,2)]))
  pred_trt <- apply(predMat, 1, which.max)
  pred_trt[pred_trt==1]=-1
  pred_trt[pred_trt==2]=1
  
  ##---------ANALYSIS OF RESUTLS---------##
  
  ## misclassification error rate
  ## compare the true optimal ITR with the estimated optimal ITR when applying to the testing data
  err<- mean(trueoptITR.test!=pred_trt)
  
  # evaluate the value function of estimated optimal ITR
  indi<-as.integer(pred_trt==Atest)
  estvalue.numerator<-mean(2*ytest*indi)
  estvalue.denominator<-mean(2*indi)
  estvalue<-estvalue.numerator/estvalue.denominator
  
  ##--------RETURN OUTPUT------------##
  return(list(bayes.neg1=bayes.neg1, bayes.pos1=bayes.pos1, pred_trt=pred_trt, err=err, estvalue=estvalue))
}




