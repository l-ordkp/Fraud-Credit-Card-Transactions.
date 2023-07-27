import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
import joblib
fake = pd.read_csv("C:\\Users\\Kshit\\Desktop\\fraudTest.csv")
# print(fake.info(),fake.head())
## Data Preprocessing 
# From the info of the data its quite obvious that columns like trans_date_trans_time ,gender,street,zip,lat,long,city_pop,dob,unix_time,merch_lat,merch_long,city,state,job won't be much valuable. So they should be removed while building a model.
fake = fake.drop(['gender','street','zip','lat','long','city_pop','dob','unix_time','merch_lat','merch_long','trans_date_trans_time','city','state','job','cc_num','trans_num'],axis=1)
# Deleting the S.No column 
fake = fake.drop(fake.columns[0],axis =1)
# print(fake.info())
# Checking number of Empty Spaces
# blank_spaces_df = fake.isna() | fake.isnull() | (fake == '')
# # Count the number of blank spaces in each column
# blank_spaces_count = blank_spaces_df.sum()
# print(blank_spaces_count)
# The number of blank spaces comes out to be zero. So there is no need for further preprocessing.

## Data Visualisation
# fake.hist(figsize=(30,30))
# plt.show()
# Pie chart

# Piechart

# Count the occurrences of each unique value in the column
# counts = fake['is_fraud'].value_counts()

# # Create a pie chart for fraud percentage
# plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
# plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.title('Fraud percentage')
# plt.show()
# Create a bar chart for number of Merchants where the transactions were fraud.
fraudmerchants = fake[fake['is_fraud']==1]
# fraudmerchantspie = fraudmerchants['merchant'].value_counts()
# plt.pie(fraudmerchantspie, labels=fraudmerchantspie.index, autopct='%1.1f%%')
# plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.title('Fraud Merchants')
# plt.show()
# This proves that there are no specific merchants through which fraud transactions occur
# fraudcategory = fraudmerchants['category'].value_counts()
# plt.pie(fraudcategory, labels=fraudcategory.index, autopct='%1.1f%%')
# plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.title('Fraud Categories')
# plt.show()

# This proves that major fraud transaction takes place through the product that categroizes as online shopping 
# fraudfirstname = fraudmerchants['first'].value_counts() 
# plt.pie(fraudfirstname, labels=fraudfirstname.index, autopct='%1.1f%%')
# plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.title('Fraud Names')
# plt.show()
# This proves that Major Fraud Transactions occur through name Christopher. 
### Machine Learning Models
## KNN
# Initialize the Label Encoder
label_encoder = LabelEncoder()

# Apply Label Encoding to 'merchant', 'category', and 'first' columns
fake['merchant'] = label_encoder.fit_transform(fake['merchant'])
fake['category'] = label_encoder.fit_transform(fake['category'])
fake['first'] = label_encoder.fit_transform(fake['first'])

# Separate features (x) and target variable (y)
x = fake[['merchant', 'category', 'amt', 'first']]
y = fake['is_fraud']


# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# # Initialize the KNN classifier with k=3 (you can choose a different value for k)
# model = KNeighborsClassifier(n_neighbors=3)

# # Train the model on the training data
# model.fit(x_train, y_train)

# # Make predictions on the test data
# y_pred = model.predict(x_test)

# # Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

#Accuracy: 0.9959422011084719

## Logistic Regression
# model1 = LogisticRegression()
# model1.fit(x_train,y_train)
# y1pred = model1.predict(x_test)
# accuracy1 = accuracy_score(y_test,y1pred)
# print("Accuracy:", accuracy1)
# Accuracy: 0.9958432304038005
## Decision Tree
# model2 = DecisionTreeClassifier()
# model2.fit(x_train,y_train)
# y2pred = model2.predict(x_test)
# accuracy2 = accuracy_score(y_test,y2pred)
# print("Accuracy:", accuracy2)
# Accuracy: 0.9962571078960628
##Random Forest Classifier
model3 = RandomForestClassifier()
# model3.fit(x_train,y_train)
# y3pred = model3.predict(x_test)
# accuracy3 = accuracy_score(y_test,y3pred)
# print("Accuracy:", accuracy3)
# Accuracy: 0.9976786871086158
##Ensemble learning with bagging

#Initialize the Bagging classifier with the base estimator and the desired number of estimators (models)
model4 = BaggingClassifier(estimator=model3, n_estimators=10, random_state=42)
# model4.fit(x_train, y_train)
# y_pred4 = model4.predict(x_test)
# accuracy4 = accuracy_score(y_test, y_pred4)
# print("Accuracy:", accuracy4)
#Accuracy: 0.9977236737925574
# Therefore we will use model4 as its accuracy is above 99%.

## Saving the machine learning model.
joblib.dump(model4, 'Ftransactiondetector.joblib')