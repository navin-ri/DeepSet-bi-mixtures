import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv("split_smiles.csv")
# Redefine key refrigerant pairs
key_test_pairs = {
    frozenset(['O=C=O', 'FC(F)F']),                        # CO₂ / R32
    frozenset(['FC=CC(F)(F)F', 'C=C(F)C(F)(F)F']),         # R1234ze(E) / R1234yf
    frozenset(['C=C(F)C(F)(F)F', 'FCC(F)(F)F'])            # R1234yf / R132a
}

# Recreate pair_key column
df['pair_key'] = df.apply(lambda row: frozenset([row['SMILES_part1'], row['SMILES_part2']]), axis=1)

# Extract forced test samples
forced_test_df = df[df['pair_key'].isin(key_test_pairs)]
remaining_df = df[~df['pair_key'].isin(key_test_pairs)]

# Sizes based on actual dataset
val_size = int(0.1 * len(df))  # ~69
test_size = int(0.1 * len(df)) - len(forced_test_df)  # remaining test samples

# Validation split
temp_train_df, val_df = train_test_split(remaining_df, test_size=val_size, random_state=42)

# Additional test samples
final_train_df, additional_test_df = train_test_split(temp_train_df, test_size=test_size, random_state=42)

# Combine forced and additional test samples
final_test_df = pd.concat([forced_test_df, additional_test_df]).reset_index(drop=True)
final_train_df = final_train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# Save the datasets to CSV
final_train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
final_test_df.to_csv("test.csv", index=False)

# Print the shapes of the splits
print(final_train_df.shape, val_df.shape, final_test_df.shape)
