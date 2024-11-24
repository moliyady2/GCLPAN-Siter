import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv


class SAGE(nn.Module):
    def __init__(self):
        # torch.manual_seed(12345)
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels=2560, out_channels=1680, aggr='mean')
        self.conv2 = SAGEConv(in_channels=1680, out_channels=640, aggr='mean')
        self.norm1 = torch.nn.BatchNorm1d(1680)

        self.lin1 = torch.nn.Linear(640, 320)
        self.lin2 = torch.nn.Linear(320, 160)
        self.lin3 = torch.nn.Linear(160, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        hid = self.conv1(x=x, edge_index=edge_index)
        hid = F.relu(hid)
        hid = self.norm1(hid)
        hid = F.dropout(hid, p=0.6, training=self.training)

        hid = self.conv2(x=hid, edge_index=edge_index)
        out = F.relu(hid)

        out = self.lin1(out)
        out = F.relu(out)

        out = self.lin2(out)
        out = F.relu(out)

        out = self.lin3(out)

        return out