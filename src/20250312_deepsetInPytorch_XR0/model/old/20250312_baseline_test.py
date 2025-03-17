import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Step 1: Load dataset
class SMILESDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop = True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        smiles_1 = self.df.loc[idx, 'SMILES_part1']
        smiles_2 = self.df.loc[idx, 'SMILES_part2']
        target = self.df.loc[idx, ['BetaT', 'GammaT', 'BetaV', 'GammaV']]
        return (smiles_1, smiles_2), torch.tensor(target, dtype = torch.float32)

# Step 2: ChemBerta Encoder
class ChemBertaEncoder(nn.Module):
    def __init__(self, pretrained_model = 'DeepChem/ChemBerta-77M-MLM'):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    def forward(self, smiles_list):
        tokens = self.tokenizer(
            smiles_list,
            padding = True,
            truncation = True,
            return_tensors = "pt"
        )

        outputs = self.model(**tokens)
        embeddings = outputs.last_hidden_state[:,0,:] #extract CLS token
        return embeddings

# step 3: DeepSet model
class DeepSet(nn.Module):
    def __init__(self, embedding_dim = 384, output_dim = 4):
        super().__init__()
        self.phi = ChemBertaEncoder()

        self.rho = nn.Sequential(
            nn.Linear(embedding_dim,128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, smiles_pairs):
        smiles_1, smiles_2 = smiles_pairs
        # Tokenize SMILES before passing to ChemBerta

        emb1 = self.phi(smiles_1)
        emb2 = self.phi(smiles_2)

        agg_emb = emb1 + emb2 # sum aggregation
        return self.rho(agg_emb)

# step 4: Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv('../split_smiles.csv')
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42)

val_df, test_df = train_test_split(test_df,
                                   test_size = 0.5,
                                   random_state)

train_df = train_df.reset_index(drop = True)
val_df = val_df.reset_index(drop = True)
test_df = test_df.reset_index(drop = True)

train_dataset = SMILESDataset(train_df)
val_dataset = SMILESDataset(val_df)
test_dataset = SMILESDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True, drop_last = True)
val_loader = DataLoader(val_dataset, batch_size = 16, shuffle = False, drop_last = False)
test_loader = DataLoader(test_dataset, batch_size= 16, shuffle= False, drop_last=False)
model = DeepSet().to(device)
criterion = nn.MSELoss()

## step 4.1: train only rho
for param in model.phi.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()),
    lr = 1e-3
)

print('\n step 1: Training only rho...')
for epoch in range(100):
    total_loss = 0
    for smiles_pairs, targets in train_loader:
        optimizer.zero_grad()
        predictions = model(smiles_pairs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {total_loss/len(train_loader):.4f}')

## Step 4.2: Unfreeze ChemBerta and fine tune all
for param in model.phi.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 1e-5
)

print('\n step 2: Fine tune all...')
for epoch in range(100):
    total_loss = 0
    for smiles_pairs, targets in train_loader:
        optimizer.zero_grad()
        predictions = model(smiles_pairs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Fine-tune Epoch {epoch + 1}, Loss: {total_loss/len(train_loader):.4f}')

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Initialize storage for each target variable
actual_values = {"BetaT": [], "GammaT": [], "BetaV": [], "GammaV": []}
predicted_values = {"BetaT": [], "GammaT": [], "BetaV": [], "GammaV": []}

target_columns = ["BetaT", "GammaT", "BetaV", "GammaV"]

# Evaluate the model on test data
model.eval()
with torch.no_grad():
    for smiles_pairs, targets in test_loader:
        predictions = model(smiles_pairs)

        # Move to CPU and convert to NumPy
        targets_np = targets.cpu().numpy()
        predictions_np = predictions.cpu().numpy()

        # Store actual and predicted values for each target
        for i, target_name in enumerate(target_columns):
            actual_values[target_name].extend(targets_np[:, i])
            predicted_values[target_name].extend(predictions_np[:, i])

# Convert lists to NumPy arrays
for key in actual_values:
    actual_values[key] = np.array(actual_values[key])
    predicted_values[key] = np.array(predicted_values[key])

# Create parity plots for each target
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for i, target_name in enumerate(target_columns):
    ax = axes[i]

    # Compute RMSE & R²
    rmse = mean_squared_error(actual_values[target_name], predicted_values[target_name], squared=False)
    r2 = r2_score(actual_values[target_name], predicted_values[target_name])

    # Scatter plot
    ax.scatter(actual_values[target_name], predicted_values[target_name], alpha=0.6, edgecolors='k')

    # 1:1 reference line
    min_val = min(actual_values[target_name].min(), predicted_values[target_name].min())
    max_val = max(actual_values[target_name].max(), predicted_values[target_name].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    # Labels & title
    ax.set_xlabel(f"Actual {target_name}")
    ax.set_ylabel(f"Predicted {target_name}")
    ax.set_title(f"Parity Plot: {target_name}")
    ax.grid(True)

    # Display RMSE & R² score on the plot
    ax.text(0.05, 0.9, f"RMSE: {rmse:.4f}\nR²: {r2:.4f}", transform=ax.transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.show()