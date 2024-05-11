import os
import pandas as pd

# Directory containing CSV files
directory = r'C:\Users\DheerajKumarJallipal\Desktop\Combined excel data'

# List to store dataframes from each CSV file
dfs = []

# Iterate through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):  # Assuming all files are CSV files
        file_path = os.path.join(directory, filename)
        # Read CSV file into a DataFrame
        df = pd.read_csv(file_path)
        # Append DataFrame to the list
        dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
final_df = pd.concat(dfs, ignore_index=True)

# Write the concatenated DataFrame to a new CSV file
final_df.to_csv(os.path.join(directory, 'final.csv'), index=False)
