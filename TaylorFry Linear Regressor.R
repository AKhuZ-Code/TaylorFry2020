# Multiple Linear Regression with Taylor Fry Case Comp

# Set working directory
getwd()
setwd("/Users/JasonKhu/Desktop/Personal Projects Folder/Consulting/Taylor Fry 2020")

# Importing the training dataset
dataset = read.csv('train_1-7.csv')
dataset = dataset[, c(2,3,4,5,6,7,8,13,14)]
dataset

# Importing the testing set
training_set = dataset
test_set = read.csv('test_1-7.csv')[, c(2,3,4,5,6,7,8,13,14)]
test_set
  # Code if splitting the dataset using R
    #install.packages('caTools')
    #library(caTools)
    #set.seed(123)
    #split = sample.split(dataset$Electricity_Demand, SplitRatio = 0.8)

# Need to create dummy variables out the Month and DayHour column
#install.packages("caret")
library(caret)
training_set$dayHour <- factor(as.character(training_set$dayHour))
training_set$Month <- factor(as.character(training_set$Month))
dmy = dummyVars(" ~ .", data=training_set)
training_set <- data.frame(predict(dmy, newdata = training_set))
training_set
training_set$dayHour.23 = NULL # to avoid dummy variable trap in training set
training_set$Month.12 = NULL

test_set$dayHour <- factor(as.character(test_set$dayHour))
test_set$Month <- factor(as.character(test_set$Month))
dmy = dummyVars(" ~ .", data=test_set)
test_set <- data.frame(predict(dmy, newdata = test_set))
test_set
test_set$dayHour.23 = NULL # to avoid dummy variable trap in test set
test_set$Month.12 = NULL

# Fitting the multiple linear regression to the training set
regressor = lm(formula = Electricity_Demand ~ ., data = training_set)
summary(regressor)
# Adjusted Rsquared = 0.6848

# Finding the optimal model
# Remove Wind_Speed variable # Adjusted Rsquared after = 0.6848
regressor = lm(formula = Electricity_Demand ~ .-Wind_Speed, data = training_set)
summary(regressor)
# Remove Air_Pressure variable # Adjusted Rsquared after = 0.6848
regressor = lm(formula = Electricity_Demand ~ .-Wind_Speed-Air_Pressure, data = training_set)
summary(regressor)
# Remove Month.2 variable # Adjusted Rsquared after = 0.6848
regressor = lm(formula = Electricity_Demand ~ .-Wind_Speed-Air_Pressure-Month.2, data = training_set)
summary(regressor) # --> Optimal model with least predictors

# Predicting the Test Set results
y_pred = predict(regressor, newdata=test_set)
y_pred
plot(y_pred, 
     main = "Prediction of Electricity Demand - Linear Regressor",
     xlab = "Days since start of 2019",
     ylab = "Electricity Demand",
     type="l",
     col="blue",
     lwd=0.5)
MSPE = mean((test_set$Electricity_Demand - predict.lm(regressor, test_set)) ^ 2)
MSPE #3596378
Root_MSPE = sqrt(MSPE)
Root_MSPE #1896.412


