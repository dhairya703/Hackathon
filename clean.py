import pandas as pd

# Load the dataset from the specified file path
file_path = 'flights_cleaned_with_direction.csv'
df = pd.read_csv(file_path)

# Define the columns that will be used to identify duplicate rows
subset_cols = ['Date', 'Sched_Departure_Time', 'Destination']

# --- Optional: Print the number of rows before deduplication ---
# initial_rows = len(df)
# print(f"Initial number of rows: {initial_rows}")

# --- Optional: Count how many duplicates are found based on the subset ---
# duplicate_rows = df.duplicated(subset=subset_cols).sum()
# print(f"Number of duplicate rows found: {duplicate_rows}")

# Remove the duplicate rows, keeping only the first occurrence of each unique combination
df_deduplicated = df.drop_duplicates(subset=subset_cols, keep='first')

# --- Optional: Print the number of rows after deduplication ---
# final_rows = len(df_deduplicated)
# print(f"Number of rows after removing duplicates: {final_rows}")

# Save the cleaned DataFrame to a new CSV file without the pandas index column
deduplicated_file_path = 'flights_deduplicated_by_subset.csv'
df_deduplicated.to_csv(deduplicated_file_path, index=False)

print(f"Successfully created a new file with duplicates removed: {deduplicated_file_path}")