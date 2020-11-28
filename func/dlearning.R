## Author: Duzhe Wang
## Last updated date: 10/10/2020
## Direct learning 




library(glmnet) ##LASSO

##########################################
#                                        #
#  D learning: when p is small           #
#                                        #
##########################################

dlearning_train_test=function(ytrain, Atrain, Xtrain, trueoptITR.test, Atest, Xtest, ytest){
  response=2*ytrain*Atrain
  
  ##linear regression 
  regdata<-data.frame(y=response, x=Xtrain)
  regmodel<-lm(y~., data=regdata)
  pred_value<-predict(regmodel, data.frame(x=Xtest))
  
  # prediction of optimal treatment 
  pred_trt<-as.numeric(pred_value>0)*2-1
  
  # calculate the misclassification rate: 
  # compare the true optimal treatment assignment with the estimated optimal treatment assignment
  err<- mean(trueoptITR.test!=pred_trt)
  
  # evaluate the value function of estimated optimal ITR
  indi<-as.integer(pred_trt==Atest)
  estvalue.numerator<-mean(2*ytest*indi)
  estvalue.denominator<-mean(2*indi)
  estvalue<-estvalue.numerator/estvalue.denominator
  
  ##--------RETURN OUTPUT------------##
  result<-list(regmodel=regmodel, pred_trt=pred_trt, err=err, estvalue=estvalue)
  return(result)
}





######################################################
#                                                    #
#  Regularized D learning: when p is large           #
#                                                    #
######################################################


## tuning 
regudlearning_validate=function(trainingX, trainingA, trainingy, valiX, valiA, valiy, lambda){
  response=2*trainingy*trainingA
  lassomodel=glmnet(trainingX, response, alpha = 1, lambda = lambda)
  pred_value=predict(lassomodel, s = lambda, newx = valiX)
  
  # prediction of optimal treatment 
  pred_trt<-as.numeric(pred_value>0)*2-1
  
  # evaluate the value function of estimated optimal ITR
  indi<-as.integer(pred_trt==valiA)
  estvalue.numerator<-mean(2*valiy*indi)
  estvalue.denominator<-mean(2*indi)
  estvalue<-estvalue.numerator/estvalue.denominator
  return(estvalue)
}

regudlearning_cvtune=function(ytrain, Atrain, Xtrain, K, lambda_seq){
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
      estvaluemat[i, k]=regudlearning_validate(trainingX, trainingA, trainingy, valiX, valiA, valiy, lambda)
    }
  }
  regudlearning_estvaluemat=cbind(estvaluemat, "means"=rowMeans(estvaluemat, na.rm=TRUE))
  regudlearning_optlambda=lambda_seq[which.max(regudlearning_estvaluemat[, "means"])]
  return(list(regudlearning_estvaluemat=regudlearning_estvaluemat, regudlearning_optlambda=regudlearning_optlambda))
}

## training and testing 

regudlearning_train_test=function(ytrain, Atrain, Xtrain, trueoptITR.test, Atest, Xtest, ytest, lambda){
  response=2*ytrain*Atrain
  trainedlasso <- glmnet(Xtrain, response, alpha = 1, lambda = lambda)
  pred_value <- predict(trainedlasso, s = lambda, newx = Xtest)
  
  
  # prediction of optimal treatment 
  pred_trt<-as.numeric(pred_value>0)*2-1
  
  # calculate the misclassification rate: 
  # compare the true optimal treatment assignment with the estimated optimal treatment assignment
  err<- mean(trueoptITR.test!=pred_trt)
  
  # evaluate the value function of estimated optimal ITR
  indi<-as.integer(pred_trt==Atest)
  estvalue.numerator<-mean(2*ytest*indi)
  estvalue.denominator<-mean(2*indi)
  estvalue<-estvalue.numerator/estvalue.denominator
  
  ##--------RETURN OUTPUT------------##
  return(list(trainedlasso=trainedlasso, coef_lasso=coef(trainedlasso), pred_trt=pred_trt, err=err, estvalue=estvalue))
}







 