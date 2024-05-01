import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from utils.graph_dataset import SMILESDataset, smiles_string,data_processed,smiles_to_one_hot
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
from sklearn.metrics import matthews_corrcoef, f1_score, cohen_kappa_score, accuracy_score, auc, roc_auc_score, average_precision_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

log_dir = '/home/zengxin/fpk/pycharm_project/GNN-DDAS/tensorboard'
writer = SummaryWriter(log_dir)

def train(train_loader, model, device, criterion, optimizer, epoch):
    model.train()
    y_true, y_pred = [], []
    loop = tqdm(enumerate(train_loader), total=len(train_loader), colour='red')
    train_loss = 0.
    logging.info(f"Training for epoch {epoch+1}")
    for step, data in loop:
        x, edge_index, smiles, labels, batch= data.x, data.edge_index, data.smiles, data.y, data.batch
        toks, smiles_mask_attn = smiles_string(smiles,291)
        toks = torch.from_numpy(np.array(toks)).to(torch.long)
        one_hot = smiles_to_one_hot(smiles,max_length=291).to(torch.float)
        smiles_mask_attn = torch.from_numpy(np.array(smiles_mask_attn)).to(torch.long)
        if torch.cuda.is_available():
            x, edge_index, toks, one_hot, smiles_mask_attn, labels, batch = x.to(device), edge_index.to(device), toks.to(device), one_hot.to(device), smiles_mask_attn.to(device), labels.to(device), batch.to(device)
        optimizer.zero_grad()
        outputs = model(x=x, edge_index=edge_index, toks=toks, one_hot=one_hot, smiles_mask_attn=smiles_mask_attn, batch=batch)
        outputs = outputs.view(-1)
        loss = criterion(outputs, labels.float()).to(device)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        loop.set_description(f'Training Epoch [{epoch} / {epochs}]')
        loop.set_postfix(loss=loss.item())
        y_true.extend(labels.cpu().bool())
        y_pred.extend(outputs.cpu())
    y_true = torch.tensor(y_true)
    y_true = y_true.view(-1).numpy()
    y_true = np.array(y_true).astype(int)
    y_pred = torch.tensor(y_pred)
    y_pred = y_pred.view(-1)
    y_pred = np.array(y_pred)
    y_pred = np.round(y_pred).astype(int)
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true,y_pred)
    AUPRC = average_precision_score(y_true,y_pred)
    f1_scores = f1_score(y_true, y_pred)
    ck = cohen_kappa_score(y_true, y_pred)
    print(f'Traing Epoch:{epoch},ACC:{acc},AUPRC:{AUPRC},f1_scores:{f1_scores},mcc:{mcc},ck:{ck}')  
    return train_loss / len(train_loader)
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
            optimizer.zero_grad()
            outputs = model(x=x, edge_index=edge_index, toks=toks, one_hot=one_hot, smiles_mask_attn=smiles_mask_attn, batch=batch)
            outputs = outputs.view(-1)
            loss = criterion(outputs, labels.float()).to(device)
            optimizer.step()
            test_loss += loss.item()
            loop.set_description(f'Test Epoch [{epoch} / {epochs}]')
            loop.set_postfix(loss=loss.item())
            y_true.extend(labels.cpu().bool())
            y_pred.extend(outputs.cpu())
    y_true = torch.tensor(y_true)
    y_true = y_true.view(-1).numpy()
    y_true = np.array(y_true).astype(int)
    y_pred = torch.tensor(y_pred)
    y_pred = y_pred.view(-1)
    y_pred = np.array(y_pred)
    y_pred = np.round(y_pred).astype(int)
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    AUPRC =average_precision_score(y_true, y_pred)
    f1_scores = f1_score(y_true, y_pred)
    ck = cohen_kappa_score(y_true, y_pred)
    test_loss /= len(test_loader)
    if test:
        print(f'Test Epoch:{epoch},acc:{acc},AUPRC:{AUPRC},f1_scores:{f1_scores},ck:{ck},mcc:{mcc}')
    else:
        print(f'Val Epoch:{epoch},acc:{acc},AUPRC:{AUPRC},f1_scores:{f1_scores},ck:{ck},mcc:{mcc}')
    return test_loss, acc, AUPRC, f1_scores, mcc, ck



if __name__ == '__main__':
     # random seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    # 数据文件路径
    train_root = '/home/zengxin/fpk/pycharm_project/GNN-DDAS/data/merge/merge_data/train_data'
    test_root = '/home/zengxin/fpk/pycharm_project/GNN-DDAS/data/merge/merge_data/test_data'
    temp_train_root = '/home/zengxin/fpk/pycharm_project/GNN-DDAS/data/merge/merge_data/resample/train'
    raw_train_root = '/home/zengxin/fpk/pycharm_project/GNN-DDAS/data/merge/merge_data/train_data/raw/train_data.csv'
    temp_test_root = '/home/zengxin/fpk/pycharm_project/GNN-DDAS/data/merge/merge_data/resample/test'
    raw_test_root = '/home/zengxin/fpk/pycharm_project/GNN-DDAS/data/merge/merge_data/test_data/raw/test_data.csv'
    save_model_path = '/home/zengxin/fpk/pycharm_project/GNN-DDAS/save_model/DL/one_label_gat'
    pretrained_model = '/home/zengxin/fpk/pycharm_project/GNN-DDAS/save_model/DL/best.pt'
    log = '/home/zengxin/fpk/pycharm_project/GNN-DDAS/log/log_one_label_gat.log'
    tensorboard = '/home/zengxin/fpk/pycharm_project/GNN-DDAS/tesnsorboard'
    
    # 超参数
    batch_size = 16
    epochs = 150
    start_epoch = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = Config.from_dict({
    "transformer_layer":1,
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
    "num_class": 2,
    "device": device
    })
    model = GNN_DDAS(config).to(device)
    
    # 加载保存好的模型
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
    
    train_set = SMILESDataset(root=train_root,raw_dataset='train_data.csv',processed_data='train.pt',max_node_num=125)
    test_set = SMILESDataset(root=test_root,raw_dataset='test_data.csv',processed_data='test.pt',max_node_num=125)
    # train_set = resampled(temp_train_root=temp_train_root,raw_train_root=raw_train_root,ratio=1)
    # test_set = resampled(temp_train_root=temp_test_root,raw_train_root=raw_test_root,ratio=1)
    learning_rate = 0.001
    weight_decay = 1e-2
    
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    bset_roc = []
    best_loss = 100.0
    best_roc = 0.0
    
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(len(train_set),len(test_set))
    count_num = 0
    max_count_num = 30
    max_num = 0
    for epoch in range(epochs):
        train_dataloader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
        test_dataloader = DataLoader(test_set,batch_size=batch_size,shuffle=False)
        train_loss = train(train_dataloader, model, device, criterion, optimizer, epoch)
        test_loss, roc, acc, auprc, f1_scores, mcc, ck = test(test_dataloader, model, device, epoch, test=True)
        if train_loss < best_loss:
            best_loss = train_loss
            count_num = 0
            _, acc, auprc, f1_scores, mcc, ck = test(test_dataloader, model, device, epoch, test=True)
            torch.save(model.state_dict(), os.path.join(save_model_path,f'merge_lr{learning_rate}_wdeacy{weight_decay}.pt'))
            with open(log, 'a') as file:
                print(f'lr:{learning_rate}, weight_decay:{weight_decay}, Epoch:{epoch},test_acc:{acc},test_AUPRC:{auprc}, test_f1:{f1_scores},test_mcc:{mcc}, test_ck:{ck}',file=file)
        else:
            count_num += 1
        # 检查是否达到最大未改善轮次
        if count_num >= max_count_num:
            print(f"No improvement for {max_count_num} epochs. Training terminated.")
            print(f"Best Epoch: {epoch},Best Learning Rate: {learning_rate},Best Decay Rate: {weight_decay},Best Loss: {best_loss}")
            break