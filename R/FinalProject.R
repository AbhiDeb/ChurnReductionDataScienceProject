rm(list = ls())

#Set Directory
setwd("D:\\edWisor\\Project-I-R")

# Importing the dataset
dataset = read.csv('Train_data.csv')
dataset_test = read.csv('Test_data.csv')

#Creating empty model performance table
performance_df = data.frame(matrix(ncol = 5, nrow = 0))
names = c("ModelName","Accuracy","True Positive Rate","Precision","Prevalence")
colnames(performance_df) = names

#LabelEncoding the data
dataset$international.plan = factor(dataset$international.plan,
                           levels = c(' no', ' yes'),
                           labels = c(0, 1))
dataset$voice.mail.plan = factor(dataset$voice.mail.plan,
                                    levels = c(' no', ' yes'),
                                    labels = c(0, 1))
dataset$Churn = factor(dataset$Churn,
                                    levels = c(' False.', ' True.'),
                                    labels = c(0, 1))

dataset_test$international.plan = factor(dataset_test$international.plan,
                                    levels = c(' no', ' yes'),
                                    labels = c(0, 1))
dataset_test$voice.mail.plan = factor(dataset_test$voice.mail.plan,
                                 levels = c(' no', ' yes'),
                                 labels = c(0, 1))
dataset_test$Churn = factor(dataset_test$Churn,
                       levels = c(' False.', ' True.'),
                       labels = c(0, 1))

#Dropping variables that are not required for the model (From the analysis done in Python)
drops <- c("phone.number","area.code",'total.eve.minutes','total.day.minutes','total.night.minutes','total.intl.minutes','state')
dataset = dataset[ , !(names(dataset) %in% drops)]
dataset_test = dataset_test[ , !(names(dataset_test) %in% drops)]

#Splitting the data - Stratified Sampling
#library(caret)
#set.seed(123)
#train.index <- createDataPartition(dataset$Churn, p = .75, list = FALSE)
#train <- dataset[ train.index,]
#test  <- dataset[-train.index,]
train = dataset
test = dataset_test


#Applying models

#1. Logistic Regression
performance_df <- PerformanceMeasureLR(train,test)
#2. KNN
performance_df <- PerformanceMeasureKNN(train,test)
#3. Naive Bayes
performance_df <- PerformanceMeasureNB(train, test)
#4. Decision Tree Classifier
performance_df <- PerformanceMeasureDT(train, test)
#5. Random Forest Classifier
performance_df <- PerformanceMeasureRF(train, test)


#Storing performance data in a csv
write.csv(performance_df, file = "Performance of models in the given Test_data (R).csv",row.names=FALSE)

replaceChurnValues <- function(a) {
  if(a==0)
    return(" False.")
  else
    return(" True.")
}

PerformanceMeasureLR <- function(train, test) {
  train_LR = train
  test_LR = test
  
  # Feature Scaling
  train_LR[1] = scale(train_LR[1])
  train_LR[4:12] = scale(train_LR[4:12])
  test_LR[1] = scale(test_LR[1])
  test_LR[4:12] = scale(test_LR[4:12])
  
  # Logistic Regression is applied to Training set
  classifier = glm(formula = Churn ~ .,
                   family = binomial,
                   data = train_LR)
  
  # Predicting the Test set results
  prob_pred = predict(classifier, type = 'response', newdata = test_LR[1:13])
  y_pred = ifelse(prob_pred > 0.5, 1, 0)
  
  # Making the Confusion Matrix
  cm = table(test_LR[, 14], y_pred)
  performance_df <- PerformanceDataFrame(performance_df,"Logistic Regression",cm)
  return(performance_df)
}

PerformanceMeasureKNN <- function(train, test) {
  train_LR = train
  test_LR = test
  
  # Feature Scaling
  train_LR[1] = scale(train_LR[1])
  train_LR[4:12] = scale(train_LR[4:12])
  test_LR[1] = scale(test_LR[1])
  test_LR[4:12] = scale(test_LR[4:12])
  
  # Fitting K-NN to the Training set and Predicting the Test set results
  library(class)
  y_pred = knn(train = train_LR[, 1:13],
               test = test_LR[, 1:13],
               cl = train_LR[, 14],
               k = 5,
               prob = TRUE)
  
  # Making the Confusion Matrix
  cm = table(test_LR[, 14], y_pred)
  performance_df <- PerformanceDataFrame(performance_df,"KNN",cm)
  return(performance_df)
}

PerformanceMeasureNB <- function(train, test) {
  train_LR = train
  test_LR = test
  
  # Fitting Naive Bayes to the Training set
  library(e1071)
  classifier = naiveBayes(x = train_LR[1:13],
                          y = train_LR$Churn)
  
  # Predicting the Test set results
  y_pred = predict(classifier, newdata = test_LR[1:13])
  
  # Making the Confusion Matrix
  cm = table(test_LR[, 14], y_pred)
  performance_df <- PerformanceDataFrame(performance_df,"Naive Bayes",cm)
  return(performance_df)
}

PerformanceMeasureDT <- function(train, test) {
  train_LR = train
  test_LR = test
  
  # Fitting Decision Tree to the Training set
  library(rpart)
  classifier = rpart(formula = Churn ~ .,
                     data = train_LR)
  
  # Predicting the Test set results
  y_pred = predict(classifier, newdata = test_LR[1:13], type = 'class')
  
  # Making the Confusion Matrix
  cm = table(test_LR[, 14], y_pred)
  performance_df <- PerformanceDataFrame(performance_df,"Decision Tree Classifier",cm)
  return(performance_df)
}

PerformanceMeasureRF <- function(train, test) {
  train_LR = train
  test_LR = test
  
  # Fitting Random Forest to the Training set
  library(randomForest)
  set.seed(123)
  classifier = randomForest(x = train_LR[1:13],
                            y = train_LR$Churn,
                            ntree = 500)
  
  # Predicting the Test set results
  y_pred = predict(classifier, newdata = test_LR[1:13])
  
  # Making the Confusion Matrix
  cm = table(test_LR[, 14], y_pred)
  performance_df <- PerformanceDataFrame(performance_df,"Decision Tree Classifier",cm)
  
  #Storing y_pred in an excel
  y_pred = data.frame(y_pred)
  names(y_pred)[names(y_pred) == "y_pred"] <- "Churn"
  y_pred$Churn = as.character(y_pred$Churn)
  y_pred$Churn[y_pred$Churn == "0"] <- ' False.'
  y_pred$Churn[y_pred$Churn == "1"] <- ' True.'
  
  write.csv(y_pred, file = "Performance of models in the given Test_data (R).csv",row.names=FALSE)
  
  return(performance_df)
}

PerformanceDataFrame <- function(df,modelname,cm) {
  print(cm)
  TN = cm[1, 1]
  TP = cm[2, 2]
  FP = cm[1, 2]
  FN = cm[2, 1]
  Total = TN + TP + FP + FN
  accuracy = (TP+TN)/Total
  sensitivity = (TP)/(TP+FN)
  precision = TP/(TP+FP)
  prevalence = (TP+FN)/Total
  df2 = data.frame("ModelName" = modelname, "Accuracy" = accuracy, "TruePositiveRate" = sensitivity, "Precision" = precision, "Prevalence" = prevalence)
  print(df2)
  df = rbind(df,df2)
  return(df)
}
