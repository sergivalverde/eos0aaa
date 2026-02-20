import os
import csv
import sys
import json

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# Paths
root = os.path.dirname(os.path.abspath(__file__))
checkpoints_dir = os.path.join(root, "..", "..", "checkpoints")
model_path = os.path.join(checkpoints_dir, "BSI_Large.pth")
params_path = os.path.join(checkpoints_dir, "BSI_Large.params.json")

# Load model params
with open(params_path) as f:
    params = json.load(f)

hidden_layers = params["hidden_layers"]
dropout = float(params["dropout"])
fp_bits = int(params["fp_bits"])


# Define model architecture (must match training)
class NeuralNetworkModel(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout_prob):
        super(NeuralNetworkModel, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Load model
model = NeuralNetworkModel(
    input_size=fp_bits,
    hidden_layers=hidden_layers,
    output_size=1,
    dropout_prob=dropout,
)
state = torch.load(model_path, map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()


def ecfp4(smi, n_bits):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


# Read input (two SMILES columns: smiles_1, smiles_2)
with open(input_file, "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = [r for r in reader]

smiles_1 = [r[0] for r in rows]
smiles_2 = [r[1] for r in rows]

# Run model
scores = []
for smi1, smi2 in zip(smiles_1, smiles_2):
    fp1 = ecfp4(smi1, fp_bits)
    fp2 = ecfp4(smi2, fp_bits)
    if fp1 is None or fp2 is None:
        scores.append(None)
    else:
        x = fp1 + fp2  # element-wise sum (same as training)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = model(x_tensor).item()
        scores.append(pred)

# Write output
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["bioactivity_similarity"])
    for s in scores:
        if s is None:
            writer.writerow([""])
        else:
            writer.writerow([round(s, 6)])
