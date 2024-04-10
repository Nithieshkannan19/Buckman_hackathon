# Buckman_hackathon

#Overview
This repository contains a Power BI project aimed at exploring and visualizing demographic distribution, employment details, and investment behavior insights from a given dataset. The dataset encompasses various attributes such as gender, marital status, age, roles, career stages, income brackets, investment behavior, knowledge levels, influencers, risk levels, and reasons for investment.This repository also contains a Python application built with Tkinter for predicting investment returns and risk levels based on user-provided information. The application utilizes machine learning models trained on investment datasets to make predictions.

#Contents
1.Power BI File
data_exploration.pbix: Power BI file containing the data visualization dashboards and reports.
2.Code Files
buckman.py: Main Python script containing the Tkinter GUI and machine learning model implementation.
Dataset_Buckman.xlsx: Sample dataset used for training machine learning models.
3.Dataset
The dataset (Dataset_Buckman.xlsx) consists of various features such as age, education, role, marital status, household income, percentage of investment, knowledge level about investment, and the target variables 'Return Earned' and 'Risk Level'.
4.Dependencies
Python 3.x
Libraries: tkinter, pandas, scikit-learn

#Features Explored
1.Demographic Distribution
Analyze the distribution of gender, marital status, and age among individuals in the dataset.
2.Employment Details
Explore employment details including roles, career stages, and income brackets represented in the dataset.
3.Investment Behavior Insights
  Investigate investment behavior insights such as:
  Percentage of household income invested
  Sources of awareness about investments
  Knowledge levels
  Influencers
  Risk levels
  Reasons for investment

#Usage
1.Installation
Ensure you have Python 3.x installed on your system.
Install the required libraries using pip install -r requirements.txt.
2.Execution
Run the investment_prediction.py script using Python.
The GUI window will appear, prompting the user to input details for prediction.
3.Input
The user needs to provide details such as age, role, education, marital status, household income, percentage of investment, and knowledge level about investment.
4.Prediction
Click the "Predict" button after entering the required details.
The application will predict the return earned and the risk level based on the provided information.
5.Output
The predicted return and risk level will be displayed in a message box.

#Model Training
#The script trains two machine learning models:
1.Random Forest Classifier for predicting investment returns.
2.Random Forest Classifier for predicting investment risk levels.

Screenshots and detailed explaination is given in the attached ppt.
