# binary mixture in Chemberta and deep set neural network
- Using ChemBerta to generate embeddings of SMILES sepatately and using the aggregate of the embeddings for final prediction MLP. 
- All these steps are tunable: 
input1/2 -> [shared transformation layer <-> aggregation function <-> prediction MLP] -> common output
---
## preprocessing
- XR0 only model dataset was used
- ChemBerta 77MLM default embedding extraction

## model
---
### chemberta -> aggregate+ MLP
deep_set_without shared.ipynb
- the best performing? 

### chemberta -> shared transformation + MLP
deep_set.ipynb
- second best model, may perfom better if chemberta was fine-tuned

### chemberta + shared transformation + MLP
chemberta_deepset.ipynb

### [ChemBerta + Shared transformation + MLP] Pytorch execution 
Pytorch_deepset.py
- Failed, with multiple bugs. 
- Need to write simple code for debugging instead of production level code.
