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

    def forward(self, x, edge_index, batch):
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
class explainer(nn.Module):
    def __init__(self, config):
        super(explainer, self).__init__()
        self.config = config
        self.smiles_gnn = gat(self.config)
        self.fc_layer = nn.Sequential(nn.Linear(128,64),
                                      nn.Linear(64,32),
                                      nn.ReLU(),
                                      nn.Linear(32,1))
        self.sigmoid_layer = nn.Sigmoid()
    def forward(self,x,edge_index,batch):
        x = self.smiles_gnn(x,edge_index,batch)
        x = self.fc_layer(x)
        x = self.sigmoid_layer(x)
        return x
