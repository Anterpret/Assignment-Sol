library(caret)
library(ggplot2)
library(randomForest)
set.seed(1234)

#import training
train<- read.csv("C:/Users/anterpreet.singh/Downloads/pml-training.csv")

#remove the zero variability columns
train_1<- train[,-nearZeroVar(train)]

#remove columns with high percentage (above 70%) of missing value
vars<- names(train_1)
mvc <- sapply(train_1[vars], function(x) sum(is.na(x)))
mvn <- names(which(mvc >= 0.7*nrow(train_1)))
vars<- setdiff(vars, mvn)
train_2<- train_1[,vars] #total 59 variables left

#remove id columns i.e X (row number), user name, timestamp and date column in the data
train_3<- train_2[,-c(1:5)]
final_predictors<- names(train_3)

#divide into training and testing
intrain<- createDataPartition(y=train_3$classe, p=0.7, list = FALSE)

training<- train_3[intrain,]
testing<- train_3[-intrain,]

#find corelations
M<- abs(cor(training[,-54]))
diag(M) <- 0
which(M > 0.7, arr.ind=T) #many variables have strong correlation

#PCA to reduce correlated variables
preProc <- preProcess(training[, -54], method = "pca", thresh = 0.95)
trainPC <- predict(preProc, training[, -54])
testPC <- predict(preProc, testing[, -54])


#run RandomForest model with cross fold validations k=4
rfMod <- train(training$classe ~ ., method = "rf", data = trainPC, 
               trControl = trainControl(method = "cv",number = 4), importance = TRUE)


#testing model results on in-sample data
pred_valid_rf <- predict(rfMod, testPC)
confus <- confusionMatrix(testing$classe, pred_valid_rf)


#import OOS test data
oos_test<- read.csv("C:/Users/anterpreet.singh/Downloads/pml-testing.csv")

#make sure only relevant columns are their in test data
oos_test_var<- names(oos_test)
oos_test_final<- intersect(oos_test_var,final_predictors)
oos_data<-oos_test[,oos_test_final] 

#create PC's using the training preproc object
pred_oos_data<- predict(preProc,oos_data)

#get final predictions on 20 test samples
pred_oos <- predict(rfMod, pred_oos_data)
pred_oos




