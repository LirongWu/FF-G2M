import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, GATConv, SAGEConv, APPNPConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, feats):
        h = feats
        h_list = []

        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                h = F.relu(h)
                h = self.dropout(h)
            h_list.append(h)

        return h_list, h


class GCN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio, activation):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(GraphConv(input_dim, output_dim, activation=activation))
        else:
            self.layers.append(GraphConv(input_dim, hidden_dim, activation=activation))
            for i in range(num_layers - 2):
                self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=activation))
            self.layers.append(GraphConv(hidden_dim, output_dim))

    def forward(self, g, feats):
        h = feats
        h_list = []

        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = self.dropout(h)
            h_list.append(h)
            
        return h_list, h


class GAT(nn.Module):
    def __init__(
        self,num_layers, input_dim, hidden_dim, output_dim, dropout_ratio, activation, num_heads=8, attn_drop=0.3, negative_slope=0.2, residual=False):
        super(GAT, self).__init__()
        hidden_dim //= num_heads
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        heads = ([num_heads] * num_layers) + [1]

        self.layers.append(GATConv(input_dim, hidden_dim, heads[0], dropout_ratio, attn_drop, negative_slope, False, activation))
        for l in range(1, num_layers - 1):
            self.layers.append(GATConv(hidden_dim * heads[l-1], hidden_dim, heads[l], dropout_ratio, attn_drop, negative_slope, residual, activation))
        self.layers.append(GATConv(hidden_dim * heads[-2], output_dim, heads[-1], dropout_ratio, attn_drop, negative_slope, residual, None))

    def forward(self, g, feats):
        h = feats
        h_list = []

        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = h.flatten(1)
            else:
                h = h.mean(1)
            h_list.append(h)

        return h_list, h


class APPNP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio, activation, edge_drop=0.5, alpha=0.1, k=10):

        super(APPNP, self).__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_ratio)

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, feats):
        h = feats
        h_list = []

        h = self.activation(self.layers[0](self.dropout(h)))
        for l, layer in enumerate(self.layers[1:-1]):
            h = self.activation(layer(h))
        h = self.layers[-1](self.dropout(h))
        
        h = self.propagate(g, h)
        h_list.append(h)

        return h_list, h


class GraphSAGE(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio, activation):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.activation = activation

        if num_layers == 1:
            self.layers.append(SAGEConv(input_dim, output_dim, aggregator_type='gcn'))
        else:
            self.layers.append(SAGEConv(input_dim, hidden_dim, aggregator_type='gcn'))
            for i in range(num_layers - 2):
                self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type='gcn'))
            self.layers.append(SAGEConv(hidden_dim, output_dim, aggregator_type='gcn'))

    def forward(self, g, feats):
        h = feats
        h_list = []

        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)
            h_list.append(h)
            
        return h_list, h


# Implementation for teacher GNNs and student MLPs
class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        if param['distill_mode'] == 0:
            self.model_name = param["teacher"]
        else:
            self.model_name = param["student"]

        if "MLP" in self.model_name:
            self.encoder = MLP(num_layers=param["num_layers"], input_dim=param["feat_dim"], hidden_dim=param["hidden_dim"], output_dim=param["label_dim"], dropout_ratio=param["dropout_t"]).to(device)
        elif "GCN" in self.model_name:
            self.encoder = GCN(num_layers=param["num_layers"], input_dim=param["feat_dim"], hidden_dim=param["hidden_dim"], output_dim=param["label_dim"], dropout_ratio=param["dropout_s"], activation=F.relu).to(device)
        elif "GAT" in self.model_name:
            self.encoder = GAT(num_layers=param["num_layers"], input_dim=param["feat_dim"], hidden_dim=param["hidden_dim"], output_dim=param["label_dim"], dropout_ratio=param["dropout_s"], activation=F.relu, num_heads=param['num_heads']).to(device)
        elif "SAGE" in self.model_name:
            self.encoder = GraphSAGE(num_layers=param["num_layers"], input_dim=param["feat_dim"], hidden_dim=param["hidden_dim"], output_dim=param["label_dim"], dropout_ratio=param["dropout_s"], activation=F.relu).to(device)
        elif "APPNP" in self.model_name:
            self.encoder = APPNP(num_layers=param["num_layers"], input_dim=param["feat_dim"], hidden_dim=param["hidden_dim"], output_dim=param["label_dim"], dropout_ratio=param["dropout_s"], activation=F.relu).to(device)


    def forward(self, data, feats):
        if "MLP" in self.model_name:
            return self.encoder(feats)[1]
        else:
            return self.encoder(data, feats)[1]


    def edge_distribution_high(self, edge_idx, feats, tau):

        src = edge_idx[1][0]
        dst = edge_idx[1][1]

        feats_abs = torch.abs(feats[src] - feats[dst])
        e_softmax = F.log_softmax(feats_abs / tau, dim=-1)

        return e_softmax

    def edge_distribution_low(self, edge_idx, feats, out, criterion_t):

        src = edge_idx[0][0]
        dst = edge_idx[0][1]

        loss = criterion_t(feats[src], out[dst])

        return loss