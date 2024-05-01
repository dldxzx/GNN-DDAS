from typing import Callable, List, Optional, Tuple, Union
from torch_geometric.data import InMemoryDataset, Data, DataLoader
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import numpy as np
import torch.nn as nn
import torch
import torch.nn as nn
from tqdm import tqdm
from Bio import SeqIO
from torch_geometric.nn import GCNConv

# SMILES原子格式
smiles_dict = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
        "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
        "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
        "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
        "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
        "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
        "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
        "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64,"*":65}

# 节点格式
atom_list = ['C', 'H', 'O', 'N', 'P', 'S', 'Cl', 'I', 'Mg', 'Se', 'F', 'As', 'Fe', 'Na', 'Br', 'Cu',  'Os', 'Co','Mo', 'R', '*','Hg','Au','Sb','Si','B','Cd','Pt',
             'Ca']
def generate_ecfp6_fingerprint(smiles):
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
    mol = Chem.MolFromSmiles(smiles)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 6, nBits=1024)  # 使用Morgan算法，半径为3，生成ECFP6
    fingerprint = fingerprint.ToBitString()
    return fingerprint
# 生成ECFP6分子指纹 
def data_processed(smiles):
    fingerprint_list = []
    for idx, smiles in enumerate(smiles):
        fingerprint = generate_ecfp6_fingerprint(smiles)
        # 将二进制字符串转换为整数
        integer_value = int(fingerprint, 2)
        # 将整数转换为 NumPy 数组
        fingerprint = np.array([int(bit) for bit in bin(integer_value)[2:].zfill(len(fingerprint))])
        fingerprint_list.append(fingerprint)
    return fingerprint_list


def smiles_onehot(smiles=None):
    smiles_one_hot = np.zeros((len(smiles),65))
    for i, amino_acid in enumerate(smiles):
        smiles_one_hot[i, smiles_dict[amino_acid]] = 1
    return smiles_one_hot.tolist()

def smiles_string(data, max_len):
    toks_list = []
    mask_attn_list = []
    for smiles in data:
        toks = [smiles_dict[char] for char in smiles]
        if len(toks) > max_len:
            toks = toks[:max_len]
            mask_attn = [1]*max_len
        else:
            toks = toks + [0] * (max_len - len(toks))
            mask_attn = [1] * len(smiles) + [0] * (max_len - len(smiles))
        toks_list.append(toks)
        mask_attn_list.append(mask_attn)
    return toks_list,mask_attn_list

def seq_smi_onehot(data, max_len):
    feature_list = []
    mask_list = []
    for idx,seq in enumerate(data):
        if max_len == 256:
            feature = smiles_onehot(seq)
            if len(feature) > 256:
                feature = feature[:256]
            mask=np.zeros(max_len)
            feature_list.append(feature)
            mask_list.append(mask)
        else:
            print('max length error!')
    for i in range(len(feature_list)):
        if len(feature_list[i]) != max_len:
            for j in range(max_len - len(feature_list[i])):
                if max_len == 1018:
                    temp = [0] * 25
                elif max_len == 256:
                    temp = [0] * 65
                feature_list[i].append(temp)
        mask_list[i][:len(feature_list[i])]=1
    return torch.from_numpy(np.array(feature_list, dtype=np.float32)),torch.from_numpy(np.array(mask_list, dtype=np.float32))
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]
def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]
def atom_features(atom,
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=True):
    if bool_id_feat:
        # return np.array([atom_to_id(atom)])
        pass
    else:
        results = one_of_k_encoding_unk(
          atom.GetSymbol(),
          [
            'B',
            'C',
            'N',
            'O',
            'F',
            'Si',
            'P',
            'S',
            'Cl',
            'As',
            'Se',
            'Br',
            'Te',
            'I',
            'At',
            'other'
          ]) + one_of_k_encoding(atom.GetDegree(),
                                 [0, 1, 2, 3, 4, 5]) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2,'other'
                  ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if not explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False
                                     ] + [atom.HasProp('_ChiralityPossible')]

        return np.array(results)


def bond_features(bond, use_chirality=True):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats)

def smiles_to_one_hot(smiles_sequence,max_length=291):
    num_chars = len(smiles_dict)
    one_hot_batch = []
    for smiles in smiles_sequence:
        one_hot = torch.zeros((max_length, num_chars), dtype=torch.int8)
        # 截断或填充 SMILES 序列
        if len(smiles) > max_length:
            smiles = smiles[:max_length]
        else:
            smiles += ' ' * (max_length - len(smiles))
        
        for i, char in enumerate(smiles):
            if char in smiles_dict:
                index = smiles_dict[char]
                # 注意此处的索引范围为 [0, num_chars-1]
                one_hot[i, index - 1] = 1
            else:
                pass
        
        one_hot_batch.append(one_hot)
    
    return torch.stack(one_hot_batch)

class SMILESDataset(InMemoryDataset):
    def __init__(self, root, raw_dataset=None, processed_data=None, max_node_num=125, transform = None, pre_transform = None):
        self.root=root
        self.raw_dataset=raw_dataset
        self.processed_data=processed_data
        self.max_protein_len=1018
        self.max_node_num=max_node_num
        self.protein_dict_len=25
        self.smiles_dict_len=65
        super(SMILESDataset,self).__init__(root, transform, pre_transform)
        self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    
    # 原始文件位置
    @property
    def raw_file_names(self):
        return [self.raw_dataset]
    
    # 文件保存位置
    @property
    def processed_file_names(self):
        return [self.processed_data]
        # return []
    
    def download(self):
        pass
    
    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            
    def process(self):
        # 读取CSV文件
        df = pd.read_csv(self.raw_paths[0])
        smiles_list = df['SMILES'].tolist()
        labels_list = df = df['label'].tolist()
        data_list = []
        for idx,smiles in enumerate(smiles_list):
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
            mol = Chem.MolFromSmiles(smiles)
            # mol = AllChem.RemoveHs(mol)
            if mol is not None:
                
                TorF = {True:1,False:0}
                # 键的格式
                bond_type = {
                'SINGLE': [1, 0, 0, 0, 0],
                'DOUBLE': [0, 1, 0, 0, 0],
                'TRIPLE': [0, 0, 1, 0, 0],
                'AROMATIC': [0, 0, 0, 1, 0],
                'IONIC': [0, 0, 0, 0, 1]}
                mol = Chem.MolFromSmiles(smiles)
                # 创建节点特征、边列表、边属性特征
                node_features = []
                edge_index = []
                edge_attr = []

                # 添加原子节点和特征
                for atom in mol.GetAtoms():
                    '''
                    f_1:原子电荷
                    f_2:原子的度
                    f_3:原子是否芳香
                    f_4:是否在环上
                    '''
                    a_1 = atom.GetFormalCharge()
                    a_2 = atom.GetDegree()
                    a_3 = atom.GetIsAromatic()
                    a_4 = atom.IsInRing()
                    f_1234 = [a_1,a_2,TorF[a_3],TorF[a_4]]
                    
                    atom_feature = [0]*(len(atom_list)+4)
                    atom_feature[atom_list.index(atom.GetSymbol())] = 1
                    atom_feature[-4:] = f_1234
                    node_features.append(atom_feature)

                # 添加键和键属性
                for bond in mol.GetBonds():
                    '''
                    b_1:键的类型（单、双）
                    b_2:键是否在环
                    '''
                    bond_feature = [0]*6
                    b_1 = bond.GetBondType()
                    b_2 = bond.IsInRing()
                    bond_feature[:5] = bond_type[str(b_1)]
                    bond_feature[-1] = TorF[b_2]
                    edge_attr.append(bond_feature)
                    edge_attr.append(bond_feature)
                    # 边
                    start_atom_idx = bond.GetBeginAtomIdx()
                    end_atom_idx = bond.GetEndAtomIdx()
                    edge_index.append([start_atom_idx, end_atom_idx])
                    edge_index.append([end_atom_idx, start_atom_idx])
                
                # 将数据转换为PyG格式
                x = torch.tensor(node_features,  dtype=torch.float)
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                if edge_index.numel() == 0:
                    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                edge_attr = torch.tensor(edge_attr,  dtype=torch.long)
                lable=labels_list[idx]
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,y=lable,smiles=smiles)
                # print(data.protein_emb.shape)
                data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data,slices),self.processed_paths[0]) 

if __name__ == '__main__':
    dataset = SMILESDataset(root='root',raw_dataset='train_data.csv',processed_data='train.pt',max_node_num=125)