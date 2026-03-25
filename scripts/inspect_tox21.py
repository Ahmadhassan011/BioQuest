import pandas as pd

# Load the pickle file
df = pd.read_pickle("data/raw/Tox21.pkl")

# Print the first 5 rows
print(df.head())
