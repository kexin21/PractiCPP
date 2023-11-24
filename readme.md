# PractiCPP
PractiCPP is a specialized tool designed for the identification of cell-penetrating peptides (CPPs), specifically tailored to tackle the challenges posed by extremely imbalanced data. To utilize PractiCPP effectively, users must prepare three distinct files based on their peptide fasta files: an ESM-2 pretrained embeddings file, a Morgan fingerprint file, and a peptide label file.

## Generating the Required Files
### ESM-2 Pretrained Embedding
For ESM-2 pretrained embeddings, we employ the `esm2_t33_650M_UR50D` model from the Facebook Research's ESM repository. This model is adept at generating peptide embeddings, particularly focusing on extracting the last layer's embeddings through mean aggregation. For detailed instructions and further information, please visit: [Facebook Research ESM](http://github.com/facebookresearch/esm).

### Morgan Fingerprint
Morgan fingerprints are generated using the RDKit tool. Below is a Python snippet demonstrating how to compute the Morgan fingerprint for a given peptide sequence:
```python
from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.MolFromFASTA(sequence)
fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
```
This script converts a peptide sequence into a molecular representation and then computes its Morgan fingerprint.

ID-Target File
This is a CSV file containing peptide IDs and their corresponding target labels. The labels are binary, where '0' denotes non-CPPs and '1' signifies CPPs.

## Hyperparameter Tuning
When dealing with imbalanced data, the hyperparameters K and N require tuning. For balanced datasets, a starting point can be setting both K and N to 1.

