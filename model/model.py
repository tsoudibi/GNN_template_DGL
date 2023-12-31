import os
import dgl
os.environ["DGLBACKEND"] = "pytorch"
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import GraphConv


class GCN_node_classification(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        '''
        Args:
            in_feats: number of input features
            h_feats: number of hidden features
            num_classes: number of output classes
        '''
        super(GCN_node_classification, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
    

class GCN_graph_classification(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        '''
        Args:
            in_feats: number of input features
            h_feats: number of hidden features
            num_classes: number of output classes
        '''
        super(GCN_graph_classification, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata["h"] = h
        # aggregate all node representations to become the graph representation
        return dgl.mean_nodes(g, "h")