import pandas as pd

# Load dataset
df = pd.read_csv("split_smiles.csv")

# Define the problematic SMILES that caused batch size issues
bad_smiles_list = ['FC(F)F', 'FC(C(F)(F)F)C(F)(F)F', 'O=O', 'Cc1ccccc1']

# Remove rows where either SMILES_part1 or SMILES_part2 is problematic
df_filtered = df[~df["SMILES_part1"].isin(bad_smiles_list) & ~df["SMILES_part2"].isin(bad_smiles_list)]

# Save the cleaned dataset
cleaned_csv_path = "split_smiles_cleaned.csv"
df_filtered.to_csv(cleaned_csv_path, index=False)

print(f"Filtered dataset saved to {cleaned_csv_path}")