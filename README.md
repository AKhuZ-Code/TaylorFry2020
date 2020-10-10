# TaylorFry2020

Project: Taylor Fry Case Competition - RNN (Python) and Linear Regression Model (R)

Date: 04/07/2020

# Built with...

• Python ()
• R

# Motivation 

Problem Statement:

<b>(Given the datasets) "Electroland would like you to produce a short-term forecasting model to predict demand for electricity"</b>

  • For our analysis on what model to recommend to the client, we decided to compare four models, where I got to work with the RNN and Linear Regressor
  
# Summary of Code
  
  • Programmed and tested the predictive power of the RNN using a 70:30 training-test split
  
  • Created dummy variables using the date field in R for the regression model - to preserve the seasonality over time (as seen in the EDA)
  
  • Programmed and tested the predictive power of the LM using a 70:30 training-test split
  
# Summary of Results
  • Used the square root of the predictive mean squared error (PMSE) to compute the accuracies of each model
  
  • PMSE of RNN = 
  
  • PMSE of LM = 
  
  • The results of this analysis were compared to that of two other models (SARIMAX (with fourier terms) and TBATS) for our model recommendation
