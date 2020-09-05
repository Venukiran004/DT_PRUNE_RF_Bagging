my_data=read.csv("C:/Users/venuk/OneDrive/Desktop/DT_PRUNE_RF_Bagging-master/bankloan.csv")
dim(my_data)


table(my_data$default)
my_data$default=as.factor(my_data$default)
#0   1 
#517 183 


library(ROSE)
set.seed(123)
my_tree_data= ovun.sample(default ~ ., data = my_data_or, method = "over",N =1034)$data

dim(my_tree_data)
table(my_tree_data$default)
# DATA SPLIT
set.seed(123)
sample_data=sample(2,nrow(my_tree_data),replace = T,prob = c(.7,.3))
sample_data

train_data1=my_tree_data[sample_data==1,]
dim(train_data1) #733   9
test_data1=my_tree_data[sample_data==2,]
dim(test_data1) #301   9



#___________________________________________________________________________________________________


######decision tree##############


library(tree)
set.seed(3)
fit_model=tree(default~., data = train_data1)
summary(fit_model)


plot(fit_model)
text(fit_model, pretty = 0)

##18 leaf nodes


library(caret)

library(ROCR)
pred_test=predict(fit_model,newdata=test_data1,type="class")

pred_test


confusionMatrix(pred_test,test_data1$default)


#####test data
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 119  42
# 1  27 113
# 
# Accuracy : 0.7708         
# 95% CI : (0.7191, 0.817)
# No Information Rate : 0.515          
# P-Value [Acc > NIR] : < 2e-16        
# 
# Kappa : 0.5425         
# 
# Mcnemar's Test P-Value : 0.09191        
#                                          
#             Sensitivity : 0.8151         
#             Specificity : 0.7290         
#          Pos Pred Value : 0.7391         
#          Neg Pred Value : 0.8071         
#              Prevalence : 0.4850         
#          Detection Rate : 0.3953         
#    Detection Prevalence : 0.5349         
#       Balanced Accuracy : 0.7721         
#                                          
#        'Positive' Class : 0    


pred_train=predict(fit_model,newdata=train_data1,type="class")

pred_train

confusionMatrix(pred_train,train_data1$default)


# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 315  87
# 1  56 275
# 
# Accuracy : 0.8049         
# 95% CI : (0.7743, 0.833)
# No Information Rate : 0.5061         
# P-Value [Acc > NIR] : < 2e-16        
# 
# Kappa : 0.6094         
# 
# Mcnemar's Test P-Value : 0.01212        
#                                          
#             Sensitivity : 0.8491         
#             Specificity : 0.7597         
#          Pos Pred Value : 0.7836         
#          Neg Pred Value : 0.8308         
#              Prevalence : 0.5061         
#          Detection Rate : 0.4297         
#    Detection Prevalence : 0.5484         
#       Balanced Accuracy : 0.8044         
#                                          
#        'Positive' Class : 0   

per_log=prediction(as.numeric(pred_test),as.numeric(test_data1$default))

ROC_Curve=performance(per_log,"tpr","fpr")
# plot(ROC_Curve3,colorize=T)
plot(ROC_Curve, colorize=T,main="ROC curve ",ylab="TPR(sensitivity)",xlab="FPR(1-specificity)")
abline(a=0,b=1)

auc=performance(per_log,"auc")
auc=unlist(slot(auc,"y.values"))
auc    #  0.7720504
auc=round(auc,4)

auc#0.7720
legend(.6,.4, auc,title="AUC")




#_________________________________________________________________________________________________


#####pruning######

set.seed(3)
cv_prune_tree<- cv.tree(fit_model, FUN= prune.misclass)
cv_prune_tree
names(cv_prune_tree)


#  cv_prune_tree
# $size
# [1] 18   14 12   10   9   4  2  1
# 
# $dev
# [1] 173 172 185 183 193 221 227 362
# 
# $k
# [1]  -Inf   0.0   3.0   3.5   6.0   8.4  11.0 136.0
# 
# $method
# [1] "misclass"
# 
# attr(,"class")
# [1] "prune"         "tree.sequence"


prune_fir_model<- prune.misclass(fit_model, best=10)
plot(prune_fir_model)
text(prune_fir_model,pretty = 0)

prune_test<- predict(prune_fir_model, test_data1, type="class")

confusionMatrix(prune_test,test_data1$default)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 119  41
# 1  27 114
# 
# Accuracy : 0.7741          
# 95% CI : (0.7226, 0.8201)
# No Information Rate : 0.515           
# P-Value [Acc > NIR] : <2e-16          
# 
# Kappa : 0.549           
# 
# Mcnemar's Test P-Value : 0.1149          
#                                           
#             Sensitivity : 0.8151          
#             Specificity : 0.7355          
#          Pos Pred Value : 0.7438          
#          Neg Pred Value : 0.8085          
#              Prevalence : 0.4850          
#          Detection Rate : 0.3953          
#    Detection Prevalence : 0.5316          
#       Balanced Accuracy : 0.7753          
#                                           
#        'Positive' Class : 0  


prune_train<- predict(prune_fir_model, train_data1, type="class")

confusionMatrix(prune_train,train_data1$default)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 302  87
# 1  69 275
# 
# Accuracy : 0.7872          
# 95% CI : (0.7557, 0.8163)
# No Information Rate : 0.5061          
# P-Value [Acc > NIR] : <2e-16          
# 
# Kappa : 0.574           
# 
# Mcnemar's Test P-Value : 0.1735          
#                                           
#             Sensitivity : 0.8140          
#             Specificity : 0.7597          
#          Pos Pred Value : 0.7763          
#          Neg Pred Value : 0.7994          
#              Prevalence : 0.5061          
#          Detection Rate : 0.4120          
#    Detection Prevalence : 0.5307          
#       Balanced Accuracy : 0.7868          
#                                           
#        'Positive' Class : 0   

per_log1=prediction(as.numeric(prune_test),as.numeric(test_data1$default))

ROC_Curve1=performance(per_log1,"tpr","fpr")
# plot(ROC_Curve3,colorize=T)
plot(ROC_Curve1, colorize=T,main="ROC curve ",ylab="TPR(sensitivity)",xlab="FPR(1-specificity)")
abline(a=0,b=1)

auc1=performance(per_log1,"auc")
auc1=unlist(slot(auc1,"y.values"))
auc1    # 0.7752762
auc1=round(auc1,4)

auc1#0.7752
legend(.6,.4, auc1,title="AUC")






#____________________________________________________________________________________

#######bagging





library(randomForest)

set.seed(3)
bag_fit=randomForest(default~., data = train_data1,mtry=8,importance=T)
bag_fit


bag_test_predict=predict(bag_fit,test_data1,type = "class")
confusionMatrix(bag_test_predict,test_data1$default)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 117   4
# 1  29 151
# 
# Accuracy : 0.8904          
# 95% CI : (0.8495, 0.9233)
# No Information Rate : 0.515           
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.7794          
# 
# Mcnemar's Test P-Value : 2.943e-05       
#                                           
#             Sensitivity : 0.8014          
#             Specificity : 0.9742          
#          Pos Pred Value : 0.9669          
#          Neg Pred Value : 0.8389          
#              Prevalence : 0.4850          
#          Detection Rate : 0.3887          
#    Detection Prevalence : 0.4020          
#       Balanced Accuracy : 0.8878          
#                                           
#        'Positive' Class : 0  




bag_train_predict=predict(bag_fit,train_data1,type = "class")
confusionMatrix(bag_train_predict,train_data1$default)


# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 371   0
# 1   0 362
# 
# Accuracy : 1         
# 95% CI : (0.995, 1)
# No Information Rate : 0.5061    
# P-Value [Acc > NIR] : < 2.2e-16 
# 
# Kappa : 1         
# 
# Mcnemar's Test P-Value : NA        
#                                     
#             Sensitivity : 1.0000    
#             Specificity : 1.0000    
#          Pos Pred Value : 1.0000    
#          Neg Pred Value : 1.0000    
#              Prevalence : 0.5061    
#          Detection Rate : 0.5061    
#    Detection Prevalence : 0.5061    
#       Balanced Accuracy : 1.0000    
#                                     
#        'Positive' Class : 0         
                             


per_log2=prediction(as.numeric(bag_test_predict),as.numeric(test_data1$default))

ROC_Curve2=performance(per_log2,"tpr","fpr")
# plot(ROC_Curve3,colorize=T)
plot(ROC_Curve2, colorize=T,main="ROC curve ",ylab="TPR(sensitivity)",xlab="FPR(1-specificity)")
abline(a=0,b=1)

auc2=performance(per_log2,"auc")
auc2=unlist(slot(auc2,"y.values"))
auc2    # 0.8877817
auc2=round(auc2,4)

auc2#0.8877
legend(.6,.4, auc2,title="AUC")





#____________________________________________________________________________________________


####random forest

set.seed(3)
randomforest_fit=randomForest(default~., data = train_data1,importance=T)
randomforest_fit

random_forest_predicted=predict(randomforest_fit,test_data1,type = "class")
confusionMatrix(random_forest_predicted,test_data1$default)


# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 120   4
# 1  26 151
# 
# Accuracy : 0.9003          
# 95% CI : (0.8608, 0.9317)
# No Information Rate : 0.515           
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.7996          
# 
# Mcnemar's Test P-Value : 0.000126        
#                                           
#             Sensitivity : 0.8219          
#             Specificity : 0.9742          
#          Pos Pred Value : 0.9677          
#          Neg Pred Value : 0.8531          
#              Prevalence : 0.4850          
#          Detection Rate : 0.3987          
#    Detection Prevalence : 0.4120          
#       Balanced Accuracy : 0.8981          
#                                           
#        'Positive' Class : 0 

random_forest_predicted_train=predict(randomforest_fit,train_data1,type = "class")

confusionMatrix(random_forest_predicted_train,train_data1$default)


# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 371   0
# 1   0 362
# 
# Accuracy : 1         
# 95% CI : (0.995, 1)
# No Information Rate : 0.5061    
# P-Value [Acc > NIR] : < 2.2e-16 
# 
# Kappa : 1         
# 
# Mcnemar's Test P-Value : NA        
#                                     
#             Sensitivity : 1.0000    
#             Specificity : 1.0000    
#          Pos Pred Value : 1.0000    
#          Neg Pred Value : 1.0000    
#              Prevalence : 0.5061    
#          Detection Rate : 0.5061    
#    Detection Prevalence : 0.5061    
#       Balanced Accuracy : 1.0000    
#                                     
#        'Positive' Class : 0   


per_log3=prediction(as.numeric(random_forest_predicted),as.numeric(test_data1$default))

ROC_Curve3=performance(per_log3,"tpr","fpr")
# plot(ROC_Curve3,colorize=T)
plot(ROC_Curve3, colorize=T,main="ROC curve ",ylab="TPR(sensitivity)",xlab="FPR(1-specificity)")
abline(a=0,b=1)

auc3=performance(per_log3,"auc")
auc3=unlist(slot(auc3,"y.values"))
auc3    # 0.8980557
auc3=round(auc3,4)

auc3#0.8981
legend(.6,.4, auc3,title="AUC")

###accuracy




decision_tree=0.7708  
prune_tree= 0.7741 
bagging=0.8904 
random_forest=0.9003 
bar_data=c(decision_tree,prune_tree,bagging,random_forest)
bar_data_names=c("DT(70)","PT(77.4)","BA(89)","RF(90)")
sorted_data=sort(bar_data)
barplot(sorted_data,names.arg=bar_data_names,col=c("grey","pink","blue","yellow"),main="ACCURACY PLOT")


###AUC
bar_auc=c(auc,auc1,auc2,auc3)
bar_auc=sort(bar_auc)
bar_auc
bar_auc_data=c("DT(77)","PT(77.5)","BA(88)","RF(89)")
 barplot(bar_auc,names.arg=bar_auc_data,,col=c("black","red","orange","green"),main="AREA UNDER CURVE")



