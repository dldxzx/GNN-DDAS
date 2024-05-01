import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool as gap, GCNConv
from transformers import BertConfig    

class Config(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(
        self,
        s_hidden_size=767
    ):
        self.s_hidden_size = s_hidden_size
    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config


    
class gat(nn.Module):
    def __init__(self, config):
        super(gat, self).__init__()
        self.config=config
        heads = self.config.gnn_head
        # self.emb = nn.Linear(self.config.gnn_input_dim,self.config.gnn_hidden_dim)
        self.cconv1 = GATConv(self.config.gnn_input_dim, self.config.gnn_hidden_dim,heads=heads)
        self.cconv2 = GATConv(self.config.gnn_hidden_dim*heads, self.config.gnn_hidden_dim,heads=heads)
        self.cconv3 = GATConv(self.config.gnn_hidden_dim*heads, self.config.gnn_hidden_dim,heads=heads)
        self.norm1 = nn.LayerNorm(self.config.gnn_hidden_dim*heads)
        self.norm2 = nn.LayerNorm(self.config.gnn_hidden_dim*heads)
        self.norm3 = nn.LayerNorm(self.config.gnn_hidden_dim*heads)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.config.gnn_dropout)
        self.flat = nn.Linear(self.config.gnn_hidden_dim*heads, self.config.gnn_output_dim)

    def forward(self, x, edge_index, batch,batch_size):
        # x = self.dropout(self.emb(x))
        x = self.dropout(self.cconv1(x, edge_index))
        x = self.norm1(x)
        x = self.relu(x)

        x = self.dropout(self.cconv2(x, edge_index))
        x = self.norm2(x)
        x = self.relu(x)

        x = self.cconv3(x, edge_index)
        x = self.norm3(x)

        x = gap(x, batch)

        x = self.flat(x)
        return x
class gcn(nn.Module):
    def __init__(self, config):
        super(gcn, self).__init__()
        self.config=config
        heads = self.config.gnn_head
        # self.emb = nn.Linear(self.config.gnn_input_dim,self.config.gnn_hidden_dim)
        self.cconv1 = GCNConv(self.config.gnn_input_dim, self.config.gnn_hidden_dim*2)
        self.cconv2 = GCNConv(self.config.gnn_hidden_dim*2, self.config.gnn_hidden_dim*4)
        self.cconv3 = GCNConv(self.config.gnn_hidden_dim*4, self.config.gnn_hidden_dim*4)
        self.norm1 = nn.LayerNorm(self.config.gnn_hidden_dim*2)
        self.norm2 = nn.LayerNorm(self.config.gnn_hidden_dim*4)
        self.norm3 = nn.LayerNorm(self.config.gnn_hidden_dim*4)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.config.gnn_dropout)
        self.flat = nn.Linear(self.config.gnn_hidden_dim*4, self.config.gnn_output_dim)

    def forward(self, x, edge_index, batch,batch_size):
        # x = self.dropout(self.emb(x))
        x = self.dropout(self.cconv1(x, edge_index))
        x = self.norm1(x)
        x = self.relu(x)

        x = self.dropout(self.cconv2(x, edge_index))
        x = self.norm2(x)
        x = self.relu(x)

        x = self.cconv3(x, edge_index)
        x = self.norm3(x)

        x = gap(x, batch)

        x = self.flat(x)

        return x

from torch_geometric.nn import SAGEConv

class sage(nn.Module):
    def __init__(self, config):
        super(sage, self).__init__()
        self.config = config
        heads = self.config.gnn_head
        # self.emb = nn.Linear(self.config.gnn_input_dim,self.config.gnn_hidden_dim)
        self.cconv1 = SAGEConv(self.config.gnn_input_dim, self.config.gnn_hidden_dim*2)
        self.cconv2 = SAGEConv(self.config.gnn_hidden_dim*2, self.config.gnn_hidden_dim*4)
        self.cconv3 = SAGEConv(self.config.gnn_hidden_dim*4, self.config.gnn_hidden_dim*4)
        # self.cconv4 = SAGEConv(self.config.gnn_hidden_dim*4, self.config.gnn_hidden_dim*4)
        self.norm1 = nn.LayerNorm(self.config.gnn_hidden_dim*2)
        self.norm2 = nn.LayerNorm(self.config.gnn_hidden_dim*4)
        self.norm3 = nn.LayerNorm(self.config.gnn_hidden_dim*4)
        # self.norm4 = nn.LayerNorm(self.config.gnn_hidden_dim*4)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.config.gnn_dropout)
        self.flat = nn.Linear(self.config.gnn_hidden_dim*4, self.config.gnn_output_dim)

    def forward(self, x, edge_index, batch,batch_size):
        # x = self.dropout(self.emb(x))
        x = self.dropout(self.cconv1(x, edge_index))
        x = self.norm1(x)
        x = self.relu(x)

        x = self.dropout(self.cconv2(x, edge_index))
        x = self.norm2(x)
        x = self.relu(x)

        x = self.cconv3(x, edge_index)
        x = self.norm3(x)

        x = gap(x, batch)

        x = self.flat(x)

        return x
from torch_geometric.nn import GINConv

class GIN(nn.Module):
    def __init__(self, config):
        super(GIN, self).__init__()
        self.config = config
        heads = self.config.gnn_head
        # self.emb = nn.Linear(self.config.gnn_input_dim,self.config.gnn_hidden_dim)
        self.cconv1 = GINConv(nn.Linear(self.config.gnn_input_dim, self.config.gnn_hidden_dim*2))
        self.cconv2 = GINConv(nn.Linear(self.config.gnn_hidden_dim*2, self.config.gnn_hidden_dim*4))
        self.cconv3 = GINConv(nn.Linear(self.config.gnn_hidden_dim*4, self.config.gnn_hidden_dim*4))
        # self.cconv4 = GINConv(nn.Linear(self.config.gnn_hidden_dim*4, self.config.gnn_hidden_dim*4))
        self.norm1 = nn.LayerNorm(self.config.gnn_hidden_dim*2)
        self.norm2 = nn.LayerNorm(self.config.gnn_hidden_dim*4)
        self.norm3 = nn.LayerNorm(self.config.gnn_hidden_dim*4)
        # self.norm4 = nn.LayerNorm(self.config.gnn_hidden_dim*4)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.config.gnn_dropout)
        self.flat = nn.Linear(self.config.gnn_hidden_dim*4, self.config.gnn_output_dim)

    def forward(self, x, edge_index, batch,batch_size):
        # x = self.dropout(self.emb(x))
        x = self.dropout(self.cconv1(x, edge_index))
        x = self.norm1(x)
        x = self.relu(x)

        x = self.dropout(self.cconv2(x, edge_index))
        x = self.norm2(x)
        x = self.relu(x)

        x = self.cconv3(x, edge_index)
        x = self.norm3(x)

        x = gap(x, batch)

        x = self.flat(x)

        return x
class linear(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(linear, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class cnn(nn.Module):
    def __init__(self, config, kernel_size=3, stride=1, padding=1):
        super(cnn, self).__init__()
        self.config = config
        input_dim = self.config.cnn_input_dim
        embedding_dim = self.config.embedding_dim
        output_dim = self.config.cnn_output_dim
        self.emb = nn.Linear(embedding_dim, 128)
        self.dropout = nn.Dropout(self.config.cnn_dropout)
        
        self.layers = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 512, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Conv1d(512, output_dim, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.AdaptiveMaxPool1d(1)
        )

    def forward(self, toks):
        x = self.dropout(self.emb(toks))
        x = self.layers(x.permute(0, 2, 1)).view(-1, 128)
        # x = self.layers(x.permute(0, 2, 1))
        return x 

class GNN_DDAS(nn.Module):
    def __init__(self, config) -> None:
        super(GNN_DDAS,self).__init__()
        self.config = config
        self.smiles_gnn = gat(self.config)
        self.linear = linear(65,128,256)
        self.cnn = cnn(self.config,kernel_size=3, stride=1, padding=1)
        self.embedding = nn.Embedding(291,64)
        self.embdding_toks_layer = nn.Linear(64, 128)
        self.embdding_one_layer = nn.Linear(256,128)
        self.fc_layer = nn.Sequential(nn.Linear(128*2,128),
                                      nn.Linear(128,64),
                                      nn.Linear(64,32),
                                      nn.ReLU(),
                                      nn.Linear(32,1))
        self.sigmoid_layer = nn.Sigmoid()
    def forward(self, x, edge_index, toks, one_hot, smiles_mask_attn, batch):
        batch_size,_ = toks.shape
        graph_emb = self.smiles_gnn(x,edge_index,batch,batch_size)
        one_hot_emb = self.linear(one_hot)
        one_hot_emb = one_hot_emb.mean(dim=1)
        one_hot_emb = self.embdding_one_layer(one_hot_emb)

        toks_emb = self.embedding(toks)
        toks_emb = toks_emb.mean(dim=1)
        toks_emb = self.embdding_toks_layer(toks_emb)
        smiles_emb = one_hot_emb+toks_emb
        fusion = torch.cat((graph_emb,smiles_emb),dim=1)
        fusion = self.fc_layer(fusion)
        fusion = self.sigmoid_layer(fusion)
        return fusion
