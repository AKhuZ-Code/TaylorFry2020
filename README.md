# TaylorFry2020

Project: Taylor Fry Case Competition - RNN (Python) and Linear Regression Model (R)

Date: 04/07/2020

## Built with...

• Python (pandas, numpy, pyplot (matplotlib), tensorflow, keras)

• R (caret)

## File Dictionary

• <b>TaylorFry RNN.py</b>: Python code used for programming/implementing the RNN

• <b>TF2020 Actual vs Predicted in Python.png</b>: Time series comparison - actual vs predicted values

• <b>TaylorFry Linear Regressor.R</b>: R code used for programming/implementing the Multiple Linear Regression model

• <b>TF2020 Actual Results in R.png:</b> Time series plot showing the actual values for Electricity Demand from the testing set - shown in R

• <b>TF2020 Predicted Results in R.png:</b> Time series plot showing the predicted values (from the Linear Regressor) using the testing set

• <b>train_1-7.csv:</b> Dataset used for training both models (csv file)

• <b>test_1-7.csv:</b> Dataset used for testing both models (csv file)

## Motivation 

Problem Statement:

<b>(a) "Electroland would like you to produce a short-term forecasting model to predict demand for electricity"</b>

<b>(b) "Electroland would like you to provide guidance on how you expect demand for electricity over the next 20 years to change"</b>

  • For our analysis on what model to recommend to the client, we decided to compare four models, where I got to work with the RNN and Linear Regressor
  
## Summary of Code
  
  • Programmed and tested the predictive power of the RNN using a ~70:30 training-test split
  
  • Created dummy variables out of the dayHour and Month field in R for the regression model - as to preserve the seasonality over time (as seen in the EDA)
  
  • Programmed and tested the predictive power of the LM using a ~70:30 training-test split
  
## Summary of Results
  • Used the square root of the predictive mean squared error (PMSE) to compute the accuracies of each model
  
  • PMSE of RNN = (386.43)^2 = 
  
  • PMSE of LM = (1896.412)^2 = 3596378
  
  • The results of this analysis were compared to that of two other models (SARIMAX (with fourier terms) and TBATS) for our model recommendation
