# Fraud-Credit-Card-Transactions.

Fake Credit Card Transactions Detection
Credit Card

Introduction
This project aims to detect fake credit card transactions using machine learning algorithms. We analyze credit card transactions data to build a predictive model that can identify potentially fraudulent transactions and help prevent unauthorized activities.

Dataset
The dataset used for this project contains historical credit card transactions with features like transaction amount, merchant name, transaction category, and customer details. The dataset is not included in this repository, but you can find it here.
Link :- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud, just in click in the dataset icon it will start automatic download 
Requirements
Python 3.6+
Libraries: Pandas, NumPy, scikit-learn and Scikit
I have done this project in VS Code
Jupyter Notebook is optional, for running the project interactively.
This dataset had 22 columns from credit card number to merchants latitude and longitude co-ordinates. I had to remove many of them as they were of little or no usemerchantcolumns,
as in the end it would decrease the accuracy of my machine learning model.as,
So in the end i used only 4 columns that category, amt, and first' for my data visualisation.visualizationcolumns:end,
The results of data visualisation were important and has been attached in this repository.have
Then I applied 5 Machine learning algorithms and used the model which had the highest accuracy.five
Then using joblib I saved the model. Then,
Then in the second file i applied the model by taking input from the user and model predicted the transaction was fake or not.
