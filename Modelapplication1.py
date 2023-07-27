import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

merchant = input("Enter the merchant name: ")
category = input("Enter the category name: ")
first = input("Enter the name of the person: ")
amt = float(input("Enter the amount of transaction: "))  # Convert to float

data = {
    'merchant': [merchant],
    'category': [category],
    'first': [first],
    'amt': [amt]
}

# Create a DataFrame using the input data
df = pd.DataFrame(data)

# Load the LabelEncoder used during model training
label_encoder = LabelEncoder()
# Apply Label Encoding to the categorical columns in the DataFrame
df['merchant'] = label_encoder.fit_transform(df['merchant'])
df['category'] = label_encoder.fit_transform(df['category'])
df['first'] = label_encoder.fit_transform(df['first'])

# Load the model from the file using joblib
loaded_model = joblib.load("C:\\Users\\Kshit\\Desktop\\FCredit\\Ftransactiondetector.joblib")

# Now you can use the loaded model for prediction
prediction = loaded_model.predict(df[['merchant', 'category', 'amt', 'first']])
if prediction== 0:
    print("The transaction might not be a fraud.")
else:
    print("The transaction might be a fraud.")
