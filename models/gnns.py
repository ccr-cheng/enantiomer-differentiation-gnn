import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, PointNetConv, SchNet, DimeNet, DimeNetPlusPlus
from torch_scatter import scatter

from ._base import register_model


@register_model('pointgcn')
class PointGCN(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers=3, cutoff=5.):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cutoff = cutoff

        self.embedding = nn.Embedding(100, embed_size)
        self.coord_embedding = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, embed_size),
        )
        self.gcns = nn.ModuleList([
            PointNetConv(
                nn.Sequential(
                    nn.Linear((2 * embed_size if i == 0 else hidden_size) + 3, 2 * hidden_size),
                    nn.GELU(),
                    nn.Linear(2 * hidden_size, hidden_size),
                ), nn.Sequential(
                    nn.Linear(hidden_size, 2 * hidden_size),
                    nn.GELU(),
                    nn.Linear(2 * hidden_size, hidden_size),
                ), aggr='add'
            ) for i in range(num_layers)
        ])
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.GELU(),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, z, pos, batch):
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, loop=False)
        node_feat = torch.cat([self.embedding(z), self.coord_embedding(pos)], dim=-1)
        for i, gcn in enumerate(self.gcns):
            node_feat = gcn(node_feat, pos, edge_index)
            if i != len(self.gcns) - 1:
                node_feat = F.gelu(node_feat)
        out = scatter(node_feat, batch, dim=0)
        return self.fc(out).squeeze(-1)


@register_model('schnet')
class SchNetPredictor(nn.Module):
    def __init__(self, hidden_channels, num_filters=128, num_interactions=6, num_gaussians=50, cutoff=5.0, **kwargs):
        super().__init__()
        self.schnet = SchNet(hidden_channels, num_filters, num_interactions, num_gaussians, cutoff, **kwargs)

    def forward(self, z, pos, batch):
        out = self.schnet(z, pos, batch)
        return torch.sigmoid(out).squeeze(-1)


@register_model('dimenet')
class DimeNetPredictor(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_blocks, num_bilinear,
                 num_spherical, num_radial, cutoff=5.0, **kwargs):
        super().__init__()
        self.dimenet = DimeNet(
            hidden_channels, out_channels, num_blocks, num_bilinear,
            num_spherical, num_radial, cutoff, **kwargs
        )

    def forward(self, z, pos, batch):
        out = self.dimenet(z, pos, batch)
        return torch.sigmoid(out).squeeze(-1)


@register_model('dimenetpp')
class DimeNetPPPredictor(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_blocks, int_emb_size, basis_emb_size,
                 out_emb_channels, num_spherical, num_radial, cutoff=5.0, **kwargs):
        super().__init__()
        self.dimenet = DimeNetPlusPlus(
            hidden_channels, out_channels, num_blocks, int_emb_size, basis_emb_size,
            out_emb_channels, num_spherical, num_radial, cutoff, **kwargs
        )

    def forward(self, z, pos, batch):
        out = self.dimenet(z, pos, batch)
        return torch.sigmoid(out).squeeze(-1)
