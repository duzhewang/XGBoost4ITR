## Author: Duzhe Wang 
## Last updated date: 10/10/2020
## Algorithm 2 in our paper


library(tidyr)
library(xgboost)


## Tuning 

dlearningbst_validate=function(trainingX, trainingA, trainingy, valiX, valiA, valiy, eta, rounds, treedepth){
  ## validation data matrix
  valiMat=data.frame(valiA, valiX)
  #valiMat.xgb=list()
  #valiMat.xgb$data=valiMat[, -1]
  #valiMat.xgb$label=valiMat[, 1] 
  #dvali=xgb.DMatrix(as.matrix(valiMat.xgb$data), label = as.vector(valiMat.xgb$label))
  
  ## training data matrix
  trainMat=data.frame(trainingA,trainingX)
  #trainMat.xgb=list()
  #trainMat.xgb$data=trainMat[,-1]
  #trainMat.xgb$label=trainMat[, 1]
  #dtrain=xgb.DMatrix(as.matrix(trainMat.xgb$data), label = as.vector(trainMat.xgb$label))
  
  ## training 
  bst <- xgboost(booster="gbtree", data = as.matrix(trainMat[,-1]), 
                 label = 2*trainingy*trainingA, max_depth = treedepth, eta=eta, nthread = 2, 
                 nrounds=rounds, objective = "reg:squarederror", verbose = 0)
  
  ## prediction
  pred_value<-predict(bst, as.matrix(valiMat[,-1])) # the output is f(x)
  pred_trt<-as.numeric(pred_value>0)*2-1
  
  # tuning criterion: evaluate the value function 
  indi<-as.integer(pred_trt==valiA)
  estvalue.numerator<-mean(2*valiy*indi)
  estvalue.denominator<-mean(2*indi)
  estvalue<-estvalue.numerator/estvalue.denominator
  return(estvalue)
}

dlearningbst_cvtune<-function(ytrain, Atrain, Xtrain, K, etavec, roundsvec, treedepthvec){
  # K: number of folds
  # etavec, roundsvec, treedepthvec: tuning parameters
  
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
      estvaluemat[i, k]=dlearningbst_validate(trainingX, trainingA, trainingy, valiX, valiA, valiy, eta, rounds, treedepth)
    }
  }
  
  dlearningbst_estvaluemat=cbind(estvaluemat, "means"=rowMeans(estvaluemat, na.rm=TRUE))
  dlearningbst_opteta=paramat[which.max(dlearningbst_estvaluemat[, "means"]), 1]
  dlearningbst_optrounds=paramat[which.max(dlearningbst_estvaluemat[, "means"]), 2]
  dlearningbst_optdepth=paramat[which.max(dlearningbst_estvaluemat[, "means"]), 3]
  
  ##---------Output------------------## 
  return(list(dlearningbst_estvaluemat=dlearningbst_estvaluemat, 
              dlearningbst_opteta=dlearningbst_opteta, 
              dlearningbst_optrounds=dlearningbst_optrounds,
              dlearningbst_optdepth=dlearningbst_optdepth))
}


## Training 

dlearningbst_train_test=function(ytrain, Atrain, Xtrain, trueoptITR.test, Atest, Xtest, ytest, eta, rounds, treedepth){
  
  ## testing data matrix
  testMat=data.frame(Atest, Xtest)
  #testMat.xgb=list()
  #testMat.xgb$data=testMat[, -1]
  #testMat.xgb$label=testMat[, 1] 
  #dtest=xgb.DMatrix(as.matrix(testMat.xgb$data), label = as.vector(testMat.xgb$label))
  
  ## training data matrix
  trainMat=data.frame(Atrain,Xtrain)
  #trainMat.xgb=list()
  #trainMat.xgb$data=trainMat[,-1]
  #trainMat.xgb$label=trainMat[, 1]
  #dtrain=xgb.DMatrix(as.matrix(trainMat.xgb$data), label = as.vector(trainMat.xgb$label))
  
  ## training 
  bst <- xgboost(booster="gbtree", data = as.matrix(trainMat[,-1]), 
                 label = 2*ytrain*Atrain, max_depth = treedepth, eta=eta, nthread = 2, 
                 nrounds=rounds, objective = "reg:squarederror", verbose = 0)
  
  ## prediction
  pred_value<-predict(bst, as.matrix(testMat[,-1])) # the output is f(x)
  pred_trt<-as.numeric(pred_value>0)*2-1
  
  ## misclassification error rate
  ## compare the true optimal ITR with the estimated optimal ITR when applying to the testing data
  err<- mean(trueoptITR.test!=pred_trt)
  
  # evaluate the value function of estimated optimal ITR
  indi<-as.integer(pred_trt==Atest)
  estvalue.numerator<-mean(2*ytest*indi)
  estvalue.denominator<-mean(2*indi)
  estvalue<-estvalue.numerator/estvalue.denominator
  
  ##--------RETURN OUTPUT------------##
  return(list(bst=bst, pred_trt=pred_trt, err=err, estvalue=estvalue))
}













