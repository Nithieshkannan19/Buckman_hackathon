# Buckman_hackathon

**Overview**<br>
This repository contains a **Power BI** project aimed at exploring and visualizing demographic distribution, employment details, and investment behavior insights from a given dataset. The dataset encompasses various attributes such as gender, marital status, age, roles, career stages, income brackets, investment behavior, knowledge levels, influencers, risk levels, and reasons for investment.This repository also contains a Python application built with Tkinter for predicting investment returns and risk levels based on user-provided information. The application utilizes machine learning models trained on investment datasets to make predictions.

**Contents**
1.Power BI File<br>
data_exploration.pbix: Power BI file containing the data visualization dashboards and reports.<br>
2.Code Files<br>
buckman.py: Main Python script containing the Tkinter GUI and machine learning model implementation.<br>
Dataset_Buckman.xlsx: Sample dataset used for training machine learning models.<br>
3.Dataset<br>
The dataset (Dataset_Buckman.xlsx) consists of various features such as age, education, role, marital status, household income, percentage of investment, knowledge level about investment, and the target variables 'Return Earned' and 'Risk Level'.<br>
4.Dependencies<br>
Python 3.x<br>
Libraries: tkinter, pandas, scikit-learn<br>

**Features Explored**<br>
1.Demographic Distribution<br>
Analyze the distribution of gender, marital status, and age among individuals in the dataset.<br>
2.Employment Details<br>
Explore employment details including roles, career stages, and income brackets represented in the dataset.<br>
3.Investment Behavior Insights<br>
  #Investigate investment behavior insights such as:<br>
  #Percentage of household income invested<br>
  #Sources of awareness about investments<br>
  #Knowledge levels<br>
  #Influencers<br>
  #Risk levels<br>
  #Reasons for investment<br>

**Usage**<br>
1.Installation<br>
Ensure you have Python 3.x installed on your system.<br>
Install the required libraries using pip install -r requirements.txt.<br>
2.Execution<br>
Run the investment_prediction.py script using Python.<br>
The GUI window will appear, prompting the user to input details for prediction.<br>
3.Input<br>
The user needs to provide details such as age, role, education, marital status, household income, percentage of investment, and knowledge level about investment.<br>
4.Prediction<br>
Click the "Predict" button after entering the required details.<br>
The application will predict the return earned and the risk level based on the provided information.<br>
5.Output<br>
The predicted return and risk level will be displayed in a message box.<br>

**Model Training**<br>
The script trains two machine learning models:<br>
1.Random Forest Classifier for predicting investment returns.<br>
2.Random Forest Classifier for predicting investment risk levels.<br>

Screenshots and detailed explaination is given in the attached ppt.
