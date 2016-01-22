# Machine Learning: Course Project
#### **Executive Summary**  
One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, data from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants is used to build a machine learning model to predict the manner in which they did the exercise.  

#### **Data Processing**  
Setup the working directory, load data files, load needed libraries, and set the seed for enabling reproducible results.  


```r
# Clean workspace & Set working directory
rm(list=ls())
setwd("E:/R/08 Machine Learning/Project 1")

# Load required packages
library(ggplot2)
library(lattice)
library(caret)
library(parallel)
library(doParallel)
```

```
## Loading required package: foreach
## Loading required package: iterators
```

```r
library(xtable)

set.seed(45673)

# Load Dataset
trainingcsv <- read.csv("./pml-training.csv")
quizcsv <- read.csv("./pml-testing.csv")
```

Split training CSV data into training and testing using createDataPartition in the caret package.  A 60% training / 40% test sample division was chosen since this data set is of medium sample size.


```r
trainIndex = createDataPartition(trainingcsv$classe,p=0.6,list=FALSE)
training = trainingcsv[trainIndex,]
testing = trainingcsv[-trainIndex,]

dim(trainingcsv)
dim(training)
dim(testing)

names(training)
str(training)
head(training)
summary(training)
nearZeroVar(training,saveMetrics=TRUE)
```

#### **Exploratory Data Analyses**  
Perform some exploratory data analyses to find which variables to use as predictors.  Showing one density plot as an example.


```r
densityplot(~gyros_arm_x | classe, data = training)
```

![](MachineLearning_PA_files/figure-html/unnamed-chunk-3-1.png) 

```r
#densityplot(~gyros_arm_y | classe, data = training)
#densityplot(~gyros_arm_z | classe, data = training)

#densityplot(~roll_arm | classe, data = training)
#densityplot(~pitch_arm | classe, data = training)
#densityplot(~yaw_arm | classe, data = training)
#densityplot(~total_accel_arm | classe, data = training)
```

Decided to start with the raw variables for each Razor inertial measurement unit (IMU) at each location (belt, arm, dumbbell, & forearm).  This data includes three-axes acceleration, gyroscope, and magnetometer (gyros, accel, magnet).


```r
imu_type <- c("gyros", "accel", "magnet")
imu_loc <- c("belt", "arm", "dumbbell", "forearm")
direction <- c("x", "y", "z")

train_vars1 <- vector("character")
for (i in 1:length(imu_loc)) {
    for (j in 1:length(imu_type)) {
        for (k in 1:length(direction)) {
            train_vars1 <- c(train_vars1,paste0(imu_type[j], '_', imu_loc[i], '_', direction[k]))
        }
    }
}
```

Next added in the variables for roll, pitch, yaw, and total acceleration for each IMU location.


```r
train_vars2 <- train_vars1
add_vars <- c("roll", "pitch", "yaw", "total_accel")
for (i in 1:length(imu_loc)) {
    for (j in 1:length(add_vars)) {
        train_vars2 <- c(train_vars2,paste0(add_vars[j], '_', imu_loc[i]))
    }
}
```

#### **Machine Learning**  

**Models**  
Three different models were used.  
Model 1: Random Forest (rf)  
Model 2: Boosting with trees (gbm)  
Model 3: Linear Discriminant Analysis (lda)  

Each of the three models was run with the first set of variables defined above (train_vars1).  
In addition each of the models was also run with the additional variables defined with (train_vars2).  
Cross validation was set in the caret package control parameters (trainControl), using 10 folds.

Used the makeCluster function from the doParallel package to utilize multiple CPU cores and shorten the run length.  


```r
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv",
                           number = 10,
                           allowParallel = TRUE)

modFit1 <- train(training$classe ~ ., data=training[train_vars1], method="rf", trControl = fitControl)
modFit2 <- train(training$classe ~ ., data=training[train_vars1], method="gbm", verbose=FALSE, trControl = fitControl)
modFit3 <- train(training$classe ~ ., data=training[train_vars1], method="lda", trControl = fitControl)

modFit1b <- train(training$classe ~ ., data=training[train_vars2], method="rf", trControl = fitControl)
modFit2b <- train(training$classe ~ ., data=training[train_vars2], method="gbm", verbose=FALSE, trControl = fitControl)
modFit3b <- train(training$classe ~ ., data=training[train_vars2], method="lda", trControl = fitControl)

stopCluster(cluster)
```

**Model Output**  


```r
#print(modFit1)
#print(modFit2)
#print(modFit3)
print(modFit1b)
```

```
## Random Forest 
## 
## 11776 samples
##    51 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 10598, 10598, 10599, 10598, 10599, 10599, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9909144  0.9885060  0.003396450  0.004297595
##   27    0.9904895  0.9879694  0.003068762  0.003882066
##   52    0.9813184  0.9763686  0.003123876  0.003948543
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```r
#print(modFit2b)
#print(modFit3b)
#print(modFit1$finalModel)
#print(modFit2$finalModel)
#print(modFit3$finalModel)
print(modFit1b$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.84%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3346    2    0    0    0 0.0005973716
## B   17 2253    9    0    0 0.0114085125
## C    0   12 2037    5    0 0.0082765336
## D    0    0   41 1886    3 0.0227979275
## E    0    0    2    8 2155 0.0046189376
```

```r
#print(modFit2b$finalModel)
#print(modFit3b$finalModel)
```

The "out of bag error" (OOB) for the random forest models (model 1 & 1b) is an estimate of the "out of sample error" rate because of how the model is constructed.  The OOB for model 1 is 1.72% and for model 1b that includes the additional predictors is 0.84%.  Also, as seen in the model report above, the resampling was set to: Cross-Validated (10 fold).


**Model Prediction on Training Data**  

```r
predict_train1 <- predict(modFit1,newdata=training[train_vars1])
predict_train2 <- predict(modFit2,newdata=training[train_vars1])
predict_train3 <- predict(modFit3,newdata=training[train_vars1])
predict_train1b <- predict(modFit1b,newdata=training[train_vars2])
predict_train2b <- predict(modFit2b,newdata=training[train_vars2])
predict_train3b <- predict(modFit3b,newdata=training[train_vars2])
conf_train1 <- confusionMatrix(training$classe,predict_train1)
conf_train2 <- confusionMatrix(training$classe,predict_train2)
conf_train3 <- confusionMatrix(training$classe,predict_train3)
conf_train1b <- confusionMatrix(training$classe,predict_train1b)
conf_train2b <- confusionMatrix(training$classe,predict_train2b)
conf_train3b <- confusionMatrix(training$classe,predict_train3b)
```

**Model Prediction on Testing Data**  

```r
predict_test1 <- predict(modFit1,newdata=testing[train_vars1])
predict_test2 <- predict(modFit2,newdata=testing[train_vars1])
predict_test3 <- predict(modFit3,newdata=testing[train_vars1])
predict_test1b <- predict(modFit1b,newdata=testing[train_vars2])
predict_test2b <- predict(modFit2b,newdata=testing[train_vars2])
predict_test3b <- predict(modFit3b,newdata=testing[train_vars2])
conf_test1 <- confusionMatrix(testing$classe,predict_test1)
conf_test2 <- confusionMatrix(testing$classe,predict_test2)
conf_test3 <- confusionMatrix(testing$classe,predict_test3)
conf_test1b <- confusionMatrix(testing$classe,predict_test1b)
conf_test2b <- confusionMatrix(testing$classe,predict_test2b)
conf_test3b <- confusionMatrix(testing$classe,predict_test3b)
```

**Model Summary**  
The table below shows the in sample error for the training set (1st & 2nd columns) and the out of sample error for the test set (3rd & 4th columns) for each model and both variable combinations.  


```r
model<-c("rf","gbm","lda")
conf_train <- c(1-conf_train1$overall['Accuracy'], 1-conf_train2$overall['Accuracy'], 1-conf_train3$overall['Accuracy'])*100
conf_trainb <- c(1-conf_train1b$overall['Accuracy'], 1-conf_train2b$overall['Accuracy'], 1-conf_train3b$overall['Accuracy'])*100

conf_test <- c(1-conf_test1$overall['Accuracy'], 1-conf_test2$overall['Accuracy'], 1-conf_test3$overall['Accuracy'])*100
conf_testb <- c(1-conf_test1b$overall['Accuracy'], 1-conf_test2b$overall['Accuracy'], 1-conf_test3b$overall['Accuracy'])*100

summary <- data.frame(model, conf_train, conf_trainb, conf_test, conf_testb)
summary
```

```
##   model conf_train conf_trainb conf_test conf_testb
## 1    rf   0.000000    0.000000  1.325516  0.6500127
## 2   gbm   6.997283    2.581522  9.240377  4.2569462
## 3   lda  35.945992   29.211957 36.655621 29.9770584
```

Several things to point out from the data in this table.  As expected the out of sample error is greater than the in sample error for each model.  Clearly, the random forest (rf) model has the lowest out of sample error, followed by boosting with trees (gbm), and linear discriminant analysis (lda) had the highest error. Comparing the two different predictor variable sets, the additional variables helped all three model types.  Last, the random forest out of sample error for each variable combination (1.33% & 0.65%) match closely to the OOB for these models stated above (1.72% & 0.84%). 

**Run Models on Quiz Data**  

```r
predict_quiz1 <- predict(modFit1,newdata=quizcsv[train_vars1])
predict_quiz2 <- predict(modFit2,newdata=quizcsv[train_vars1])
predict_quiz3 <- predict(modFit3,newdata=quizcsv[train_vars1])
predict_quiz1b <- predict(modFit1b,newdata=quizcsv[train_vars2])
predict_quiz2b <- predict(modFit2b,newdata=quizcsv[train_vars2])
predict_quiz3b <- predict(modFit3b,newdata=quizcsv[train_vars2])
quiz <- data.frame(quizcsv$problem_id, predict_quiz1, predict_quiz1b, predict_quiz2, predict_quiz2b, predict_quiz3, predict_quiz3b)
names(quiz) <- c("id","cl_1_rf","cl_1b_rf","cl_2_gbm","cl_2b_gbm","cl_3_lda","cl_3b_lda")
quiz
```

```
##    id cl_1_rf cl_1b_rf cl_2_gbm cl_2b_gbm cl_3_lda cl_3b_lda
## 1   1       B        B        B         B        B         B
## 2   2       A        A        A         A        A         A
## 3   3       B        B        B         B        B         B
## 4   4       A        A        A         A        C         C
## 5   5       A        A        A         A        C         C
## 6   6       E        E        E         E        E         E
## 7   7       D        D        D         D        D         D
## 8   8       B        B        B         B        D         D
## 9   9       A        A        A         A        A         A
## 10 10       A        A        A         A        A         A
## 11 11       A        B        A         B        A         D
## 12 12       C        C        C         C        A         A
## 13 13       B        B        B         B        B         B
## 14 14       A        A        A         A        A         A
## 15 15       E        E        E         E        E         E
## 16 16       E        E        E         E        A         A
## 17 17       A        A        A         A        C         A
## 18 18       B        B        B         B        B         B
## 19 19       B        B        B         B        B         B
## 20 20       B        B        B         B        B         B
```

#### **Conclusion**  
The random forest model using the larger predictor variable set (cl_1b_rf) in the quiz table above was used for the course project prediction quiz, and it correctly predicted all 20 cases.  It is interesting that the in sample error was zero for both the random forest models using the two different predictor variable sets.  However, the out of sample error was slightly lower for the additional variable predictor set.  This lower out of sample error was significant in the quiz prediction, as id case 11 was incorrectly predicted by the smaller predictor set.   
