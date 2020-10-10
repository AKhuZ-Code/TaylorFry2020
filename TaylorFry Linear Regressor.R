# Multiple Linear Regression with Taylor Fry Case Comp

# Set working directory
getwd()
setwd("/Users/JasonKhu/Desktop/Consulting/Taylor Fry 2020")

# Importing the trainign dataset
dataset = read.csv('train_1-7.csv')
dataset = dataset[, c(2,3,4,5,6,7,8,13,14)]
dataset

# Importing the testing set
# install.packages('caTools')
library(caTools)
# set.seed(123)
# split = sample.split(dataset$Electricity_Demand, SplitRatio = 0.8)
#JOKES: Import the training set and test set
training_set = dataset
test_set = read.csv('test_1-7.csv')[, c(2,3,4,5,6,7,8,13,14)]
test_set

#Fitting the multiple linear regression to the training set
regressor = lm(formula = Electricity_Demand ~ ., data = training_set)
summary(regressor)

# Predicting the Test Set results
y_pred = predict(regressor, newdata=test_set)
y_pred
plot(y_pred)
MSPE = mean((test_set$Electricity_Demand - predict.lm(regressor, test_set)) ^ 2)
MSPE
MSPE/length(y_pred)
sqrt(MSPE)

# Building the optimal model using backward elimination
regressor = lm(formula = Electricity_Demand ~ Temperature + Air_Pressure + Total_Pop + Unemployment_Rate + Solar_PV_penetration + dayHour + Season
               , data = training_set)
#Adjusted R squared is 0.3474
summary(regressor)
#RMSE is 2933.49
