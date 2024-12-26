# GCLKSiter
Pytorch implementation of paper 'GCLKSiter: ' 

## Dependencies
- torch=2.0.0+cu118
- torch-geometric=2.3.0
- torch-scatter=2.1.1+pt20cu118
- torch-sparse=0.6.17+pt20cu118
- torchvision=0.15.1+cu118
- scikit-learn
- scipy=1.10.1
- PyGCL=0.1.2
- dgl=1.0.0
- biopython
- numpy
- tqdm

## Repo for iProtDNA-SMOTE framework
This repo holds the code of iProtDNA-SMOTE framework for protein-ligands binding sites prediction. Five processed datasets are published, including TR52, TR41, TE335, TE543, TE573 and TE129.

iProtDNA-SMOTE is primarily dependent on a large-scale pre-trained protein language model ESM2 implemented using PyTorch. Please install the dependencies in advance.

## Files and folders description
### DNAPred_Dataset
This folder contains the raw data. For each protein id inside (which may not be the PDB id), there are two files that store the sequence and label information for that protein. Among them, label information is a data label indicating binding or non-binding sites.
### evaluate & train
These folders contain pre-training weights.

Note: if you wanna find the original data in PDB format, please kindly refer to the following 3 papers: DBPred, GraphBind, and GraphSite.



## Codes description
### 1. GRACE_classifier.py
The main program that generates binding site predictions and evaluates them

### 2. ESM2.py
Import pre-trained protein language model ESM2 to generate feature vector.

### 3.Distance_matrix.py

The atomic coordinates are extracted from the pdb file, then the distance matrix is generated according to the coordinates.

### 4.Dataset.py

The protein sequence data and corresponding eigenmatrix and label information are converted into graph data structure

### 5. Evaluate.py

Evaluate the model's performance on the data set.

### 6. Graphsage.py

Implementation of inductive representation learning framework Graphsage

### 7. Focal_loss.py

Implementation of loss function.

## Citation

If any problems occur via running this code, please contact us at 13507980109@163.com.

Thank you!
