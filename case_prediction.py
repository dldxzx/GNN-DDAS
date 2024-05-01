from utils.graph_dataset import SMILESDataset, smiles_string,smiles_to_one_hot
from utils.GNN_DDAS import GNN_DDAS, Config
from utils.resample import resampled
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
import os
import logging
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from sklearn.metrics import matthews_corrcoef, f1_score, cohen_kappa_score,accuracy_score, roc_curve, auc, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

log_dir = '/home/zengxin/fpk/pycharm_project/GNN-DDAS/tesnsorboard'
writer = SummaryWriter(log_dir)

def test(test_loader, model, device, epoch,test=True):
    model.eval()
    if test:
        loop = tqdm(enumerate(test_loader), total=len(test_loader), colour='green')
    else:
        loop = tqdm(enumerate(test_loader), total=len(test_loader), colour='white')
    y_true, y_pred = [], []
    test_loss = 0.
    logging.info(f"Testing")
    with torch.no_grad():
        for step, data in loop:
            x, edge_index, smiles, labels, batch= data.x, data.edge_index, data.smiles, data.y, data.batch
            toks, smiles_mask_attn = smiles_string(smiles,291)
            toks = torch.from_numpy(np.array(toks)).to(torch.long)
            one_hot = smiles_to_one_hot(smiles,max_length=291).to(torch.float)
            smiles_mask_attn = torch.from_numpy(np.array(smiles_mask_attn)).to(torch.long)
            if torch.cuda.is_available():
                x, edge_index, toks, one_hot, smiles_mask_attn, labels, batch = x.to(device), edge_index.to(device), toks.to(device), one_hot.to(device), smiles_mask_attn.to(device), labels.to(device), batch.to(device)
            outputs = model(x=x, edge_index=edge_index, toks=toks, one_hot=one_hot, smiles_mask_attn=smiles_mask_attn, batch=batch)
            outputs = outputs.view(-1)
            loop.set_description(f'Test Epoch [{epoch} / {epochs}]')
            y_pred.extend(outputs.cpu())
    y_pred = np.array(y_pred)
    y_pred = np.round(y_pred).astype(int)
    print(y_pred)
    


if __name__ == '__main__':
     # random seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    pretrained_data = '/home/zengxin/fpk/pycharm_project/GNN-DDAS/data/AASM_17/raw'
    pretrained_model = '/home/zengxin/fpk/pycharm_project/GNN-DDAS/save_model/DL/one_label_gat/best_one_label_gat_merge_lr0.001_wdeacy0.01.pt'
    batch_size = 16
    epochs = 150
    start_epoch = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = Config.from_dict({
    "transformer_layer":2,
    "num_attention_heads":8,
    "hidden_size":256,
    "cnn_input_dim": 256,
    "embedding_dim": 32,
    "cnn_dropout": 0.4,
    "cnn_output_dim": 128,
    "gnn_input_dim": 33,
    "gnn_head": 8,
    "gnn_hidden_dim":64,
    "gnn_output_dim":128,
    "pool_input_dim":128,
    "gnn_dropout": 0.4,
    "num_class": 1,
    "device": device
    })
    model = GNN_DDAS(config).to(device)
    # load pretrained model
    if os.path.exists(pretrained_model) and pretrained_model != " ":
        logging.info(f"开始加载GNN-DDAS模型")
        try:
            state_dict = torch.load(pretrained_model)
            print(state_dict)
            new_model_state_dict = model.state_dict()
            for key in new_model_state_dict.keys():
                if key in state_dict.keys():
                    try:
                        new_model_state_dict[key].copy_(state_dict[key])
                    except:
                        None
            model.load_state_dict(new_model_state_dict)
            logging.info("GNN-DDAS模型加载成功(!)")
        except:
            new_state_dict = {}
            for key, value in state_dict.items():
                new_state_dict[key.replace("module.", "")] = value
            model.load_state_dict(new_state_dict)
            logging.info("GNN-DDAS模型加载成功(!!!)")

    else:
        logging.info("模型路径不存在，不能加载模型")
    test_set = SMILESDataset(root=pretrained_data,raw_dataset='test_data.csv',processed_data='test.pt')
    test_dataloader = DataLoader(test_set,batch_size=batch_size,shuffle=False)
    test(test_dataloader, model, device, epoch=0, test=True)
