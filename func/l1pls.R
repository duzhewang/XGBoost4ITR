## Author: Duzhe Wang
## Last updated date: 10/10/2020
## l1-PLS (Qian and Murphy 2011) 
## this method needs to select the tuning parameter. We use K-fold cross validation based on estimated value.   


library(glmnet) ##LASSO

l1pls_validate=function(trainingX, trainingA, trainingy, valiX, valiA, valiy, lambda){
  ## training data matrix
  newtrainMat=data.frame(x=trainingX, a=trainingA, xa=trainingX*trainingA)
  
  ## training 
  lasso_best <- glmnet(as.matrix(newtrainMat), trainingy, alpha = 1, lambda = lambda)
  
  ## validation data matrix
  newvaliMat.pos1=data.frame(x=valiX, a=rep(1, length(valiy)), xa=valiX*rep(1, length(valiy)))
  newvaliMat.neg1=data.frame(x=valiX, a=rep(-1, length(valiy)), xa=valiX*rep(-1, length(valiy)))
  
  ## prediction
  predMat<-matrix(data=NA, nrow=nrow(valiX), ncol=2)
  predMat[,1] <- predict(lasso_best, s=lambda, newx=as.matrix(newvaliMat.neg1))
  predMat[,2] <- predict(lasso_best, s=lambda, newx=as.matrix(newvaliMat.pos1))
  pred_trt <- apply(predMat, 1, which.max)
  pred_trt[pred_trt==1]=-1
  pred_trt[pred_trt==2]=1
  
  # evaluate the value function of estimated optimal ITR
  indi<-as.integer(pred_trt==valiA)
  estvalue.numerator<-mean(2*valiy*indi)
  estvalue.denominator<-mean(2*indi)
  estvalue<-estvalue.numerator/estvalue.denominator
  return(estvalue)
}


## tuning using K-fold cross validation 

l1pls_cvtune=function(ytrain, Atrain, Xtrain, K, lambda_seq){
  # ytrain, Atrain, Xtrain: whole training data
  # K: cross validation fold
  # lambda_seq: lambda sequence
  
  # length of each fold
  foldlen<-length(ytrain)/K 
  
  estvaluemat=matrix(NA, nrow=length(lambda_seq), ncol=K)
  for(i in 1:length(lambda_seq)){
    lambda=lambda_seq[i]
    for(k in 1:K){
      index=((k-1)*foldlen+1):(k*foldlen)
      trainingX=Xtrain[-index,]
      trainingA=Atrain[-index]
      trainingy=ytrain[-index]
      valiX=Xtrain[index,]
      valiA=Atrain[index]
      valiy=ytrain[index]
      estvaluemat[i, k]=l1pls_validate(trainingX, trainingA, trainingy, valiX, valiA, valiy, lambda)
    }
  }
  l1pls_estvaluemat=cbind(estvaluemat, "means"=rowMeans(estvaluemat, na.rm=TRUE))
  l1pls_optlambda=lambda_seq[which.max(l1pls_estvaluemat[, "means"])]
  return(list(l1pls_estvaluemat=l1pls_estvaluemat, l1pls_optlambda=l1pls_optlambda))
}


## training and testing 

l1pls_train_test=function(ytrain, Atrain, Xtrain, trueoptITR.test, Atest, Xtest, ytest, lambda){
  ## training data matrix
  trainMat=data.frame(x=Xtrain, a=Atrain, xa=Xtrain*Atrain)
  
  ## testing data matrix
  testMat.pos1=data.frame(x=Xtest, a=rep(1, length(ytest)), xa=Xtest*rep(1, length(ytest)))
  testMat.neg1=data.frame(x=Xtest, a=rep(-1, length(ytest)), xa=Xtest*rep(-1, length(ytest)))
  
  ## training 
  trainedlasso <- glmnet(as.matrix(trainMat), ytrain, alpha = 1, lambda = lambda)
  
  ## prediction
  predMat<-matrix(data=NA, nrow=nrow(Xtest), ncol=2)
  predMat[,1] <- predict(trainedlasso, s=lambda, newx=as.matrix(testMat.neg1))
  predMat[,2] <- predict(trainedlasso, s=lambda, newx=as.matrix(testMat.pos1))
  pred_trt <- apply(predMat, 1, which.max)
  pred_trt[pred_trt==1]=-1
  pred_trt[pred_trt==2]=1
  
  # calculate the misclassification rate: 
  # compare the true optimal treatment assignment with the estimated optimal treatment assignment
  err<- mean(trueoptITR.test!=pred_trt)
  
  # evaluate the value function of estimated optimal ITR
  indi<-as.integer(pred_trt==Atest)
  estvalue.numerator<-mean(2*ytest*indi)
  estvalue.denominator<-mean(2*indi)
  estvalue<-estvalue.numerator/estvalue.denominator
  
  ##--------RETURN OUTPUT------------##
  result<-list(trainedlasso=trainedlasso, coef_lasso=coef(trainedlasso), pred_trt=pred_trt, err=err, estvalue=estvalue)
  return(result)
}





