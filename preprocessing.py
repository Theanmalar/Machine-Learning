import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
data = {
    'Name': ['Ragu', 'priya', 'anu', 'karan', 'Ragu', 'janu'],
    'Age': [25, np.nan, 22, 28, 30, 25],
    'Salary': [50000, 54000, 58000, np.nan, 60000, 50000],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female']
}

df = pd.DataFrame(data)
print("Original Data:\n", df)

# Remove duplicates (rows with all identical values)
df = df.drop_duplicates()

# Handle missing values: fill with column mean
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())

# Normalize numerical columns
scaler = MinMaxScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

# Encode categorical columns
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])

# Feature engineering
df['Salary_per_age'] = df['Salary'] / df['Age']

print("\nProcessed Data:\n", df)
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

sns.boxplot(y=df['Salary'])
plt.title('Salary Outliers')
plt.show()

sns.scatterplot(x=df['Age'], y=df['Salary'])
plt.title('Salary vs Age')
plt.show()
