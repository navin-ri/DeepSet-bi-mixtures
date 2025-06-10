from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd


# ChemBerta Encoder
class ChemBertaEncoder(nn.Module):
    def __init__(self, pretrained_model='DeepChem/ChemBerta-77M-MLM'):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    def forward(self, smiles_list):
        tokens = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        tokens = {k: v.to(next(self.model.parameters()).device) for k, v in tokens.items()}
        outputs = self.model(**tokens)
        embeddings = outputs.last_hidden_state[:, 0, :]

        # Print batch details if batch size is too small
        if embeddings.shape[0] != 16:
            print(f"⚠️  Batch issue detected! Expected 16, got {embeddings.shape[0]}")
            print(f"Problematic SMILES: {smiles_list}")

        return embeddings

# Deep Set neural network
class DeepSet(nn.Module):
    def __init__(self, embedding_dim=384, output_dim=4):
        super().__init__()
        self.phi = ChemBertaEncoder()

        self.rho = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    @staticmethod
    def aggregate(emb1, emb2):
        return emb1 + emb2  # sum aggregation

    def forward(self, smiles_pairs):
        smiles_1 = [pair[0] for pair in smiles_pairs]
        smiles_2 = [pair[1] for pair in smiles_pairs]

        emb1 = self.phi(smiles_1)
        emb2 = self.phi(smiles_2)

        agg_emb = self.aggregate(emb1, emb2)
        return self.rho(agg_emb)

class SMILESDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        smiles_1 = self.df.iloc[idx]['SMILES_part1']
        smiles_2 = self.df.iloc[idx]['SMILES_part2']
        targets = self.df.iloc[idx][['BetaT', 'GammaT', 'BetaV', 'GammaV']].values.astype(float)
        return (smiles_1, smiles_2), torch.tensor(targets, dtype=torch.float32)

# Data loading
dataset = SMILESDataset('split_smiles_cleaned.csv')
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model setup
model = DeepSet()
criterion = nn.MSELoss()

# Step 1: Freeze phi, train rho
for param in model.phi.parameters():
    param.requires_grad = False
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)

for epoch in range(10):
    total_loss = 0
    for smiles_pairs, targets in loader:
        optimizer.zero_grad()
        predictions = model(smiles_pairs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(loader):.4f}')

# Step 2: Unfreeze phi, fine-tune both
for param in model.phi.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(10):
    total_loss = 0
    for smiles_pairs, targets in loader:
        optimizer.zero_grad()
        predictions = model(smiles_pairs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Fine-tune Epoch {epoch + 1}, Loss: {total_loss / len(loader):.4f}')
