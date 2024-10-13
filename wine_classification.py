import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Read Data
df = pd.read_csv('WineQT.csv')

# Drop ID column as pandas dataframe already has an index
df = df.drop(columns=['Id'])

# Check Null Value
print("Check Null Value")
print(df.isnull().sum())

# Check duplicated rows
print("\nCheck Duplicate")
print(df.duplicated().sum())

# Clear duplicated rows
df = df.drop_duplicates()

# Check data info
print("\nAfter Cleaning, the current data shape:")
print(df.shape)

# Split Train and Test dataset
x = df.drop(columns=['quality'])
y = df['quality']

# Scale Inputs
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x = pd.DataFrame(x, columns=df.drop(columns=['quality']).columns)
print(x)

# x_train, x_test, y_train, y_test = train_test_split()
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# print("Training set size:", x_train.shape)
# print("Testing set size:", x_test.shape)

