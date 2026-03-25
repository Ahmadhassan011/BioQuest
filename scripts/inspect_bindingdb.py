import pandas as pd

# Load the pickle file
df = pd.read_pickle("data/raw/BindingDB.pkl")

# Print the first 5 rows
print(df.head())
