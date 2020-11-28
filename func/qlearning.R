## Author: Duzhe Wang 
## Last updated date: 10/10/2020
## Q-learning described in our paper


qlearning_train_test=function(ytrain, Atrain, Xtrain,trueoptITR.test, Atest, Xtest, ytest){
   # ytrain, Atrain, Xtrain: training data
   # trueoptITR.test: true optimal ITR applied to the testing data
   # Atest, Xtest, ytest: testing data

   ##-------MAKE DATA FRAME FOR TRAINING DATA-----------------##
   trainM <- data.frame(ytrain, Atrain, Xtrain)

   ##------MAKE DATA FRAME FOR TESTING DATA--------------------##
   testM <- data.frame(ytest, Atest, Xtest)

   ##--------TRAINING-------------------------------------------##
   neg1Mat<-data.frame(y= trainM$ytrain[trainM$Atrain==-1], x=trainM[trainM$Atrain==-1,-c(1,2)])
   pos1Mat<-data.frame(y= trainM$ytrain[trainM$Atrain==1],  x=trainM[trainM$Atrain==1,-c(1,2)])
   coef_neg1<-lm(y~., data=neg1Mat)
   coef_pos1<-lm(y~., data=pos1Mat)

   ##---------TESTING-------------------------------------------##
   pred_lm<-matrix(data=NA, nrow=nrow(Xtest), ncol=2)  # used to save predicted optimal ITR
   pred_lm[,1]<-predict(coef_neg1, data.frame(x=testM[-c(1,2)]))
   pred_lm[,2]<-predict(coef_pos1, data.frame(x=testM[-c(1,2)]))
   pred_trt <- apply(pred_lm, 1, which.max)
   ## preduction for the testing data
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
   return(list(coef_neg1=coef_neg1, coef_pos1=coef_pos1, pred_trt=pred_trt, err=err, estvalue=estvalue))
}

























