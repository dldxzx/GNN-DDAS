# GNN-DDAS

GNN-DDAS: Drug Discovery for Identifying Anti-Schistosome Small Molecules based on Graph Neural Network

## Requirements

[numpy](https://numpy.org/)==1.22.1

[pandas](https://pandas.pydata.org/)==2.0.3

[rdkit](https://www.rdkit.org/)==2023.09.1

[scipy](https://scipy.org/)==1.7.3

[torch](https://pytorch.org/)==1.12.1

[torch_geometric]([PyG Documentation — pytorch_geometric documentation (pytorch-geometric.readthedocs.io)](https://pytorch-geometric.readthedocs.io/en/latest/index.html))==2.0.2

## Example usage

### 1. Use our pre-trained model
In this section, we provide the collected Anti-Schistosome dataset. You can directly execute the following command to run our pre-trained model and obtain results.
```bash
# Run the following command.
python test_pretrain.py
```

### 2. Run on your datasets

In this section, you must provide a .csv file containing drug SMILES sequences. You can use the provided code for data preprocessing to ensure that the model runs correctly.
 ```bash
# you can use the following command-line script to process your data:
python graph_dataset.py

# When all the data is ready, you can train your own model by running the following command.
python training.py

 ```
