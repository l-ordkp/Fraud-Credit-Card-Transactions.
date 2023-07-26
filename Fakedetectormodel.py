import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fake = pd.read_csv("C:\\Users\\Kshit\\Desktop\\fraudTest.csv")
# print(fake.info(),fake.head())
## Data Preprocessing 
# From the info of the data its quite obvious that columns like trans_date_trans_time ,gender,street,zip,lat,long,city_pop,dob,unix_time,merch_lat,merch_long,city,state,job won't be much valuable. So they should be removed while building a model.
fake = fake.drop(['gender','street','zip','lat','long','city_pop','dob','unix_time','merch_lat','merch_long','trans_date_trans_time','city','state','job','cc_num','trans_num'],axis=1)
# Deleting the S.No column 
fake = fake.drop(fake.columns[0],axis =1)
print(fake.head())
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
fraudfirstname = fraudmerchants['first'].value_counts() 
plt.pie(fraudfirstname, labels=fraudfirstname.index, autopct='%1.1f%%')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Fraud Names')
plt.show()
 ## Machine Learning Models
 # Logistic Regression

