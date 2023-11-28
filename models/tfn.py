import torch
from torch import nn
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
from e3nn import o3
from e3nn.math import soft_one_hot_linspace

from ._base import register_model


class NormActivation(nn.Module):
    def __init__(self, irreps_in, act_scalars=torch.nn.functional.silu, act_vectors=torch.sigmoid):
        super(NormActivation, self).__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.scalar_irreps = self.irreps_in[0:1]
        self.vector_irreps = self.irreps_in[1:]
        self.act_scalars = act_scalars
        self.act_vectors = act_vectors
        self.scalar_idx = self.irreps_in[0].mul

        inner_out = o3.Irreps([(mul, (0, 1)) for mul, _ in self.vector_irreps])
        self.inner_prod = o3.TensorProduct(
            self.vector_irreps, self.vector_irreps, inner_out, [
                (i, i, i, 'uuu', False) for i in range(len(self.vector_irreps))
            ]
        )
        self.mul = o3.ElementwiseTensorProduct(inner_out, self.vector_irreps)

    def forward(self, features):
        scalars = self.act_scalars(features[..., :self.scalar_idx])
        vectors = features[..., self.scalar_idx:]
        norm = torch.sqrt(self.inner_prod(vectors, vectors) + 1e-8)
        act = self.act_vectors(norm)
        vectors_out = self.mul(act, vectors)
        return torch.cat([scalars, vectors_out], dim=-1)


class TFNLayer(nn.Module):
    def __init__(self, irreps_in, irreps_out, irreps_edge, radial_embed_size, radial_hidden_size):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.radial_embed_size = radial_embed_size
        self.radial_hidden_size = radial_hidden_size

        instr = [
            (i_1, i_2, i_out, 'uvu', True)
            for i_1, (_, ir_1) in enumerate(self.irreps_in)
            for i_2, (_, ir_edge) in enumerate(self.irreps_edge)
            for i_out, (_, ir_out) in enumerate(self.irreps_out)
            if ir_out in ir_1 * ir_edge
        ]
        self.tp = o3.TensorProduct(
            self.irreps_in, self.irreps_edge, self.irreps_out, instr,
            internal_weights=False, shared_weights=False,
        )
        self.fc = nn.Sequential(
            nn.Linear(radial_embed_size, radial_hidden_size),
            nn.SiLU(),
            nn.Linear(radial_hidden_size, radial_embed_size),
            nn.SiLU(),
            nn.Linear(radial_embed_size, self.tp.weight_numel),
        )
        self.sc = o3.Linear(self.irreps_in, self.irreps_out)

    def forward(self, edge_index, node_feat, edge_feat, edge_embed, dim_size=None):
        src, dst = edge_index
        weight = self.fc(edge_embed)
        out = self.tp(node_feat[src], edge_feat, weight=weight)
        out = scatter(out, dst, dim=0, dim_size=dim_size, reduce='sum')
        return out + self.sc(node_feat)


@register_model('tfn')
class TFNPredictor(nn.Module):
    def __init__(self, num_radial, num_spherical, radial_embed_size, radial_hidden_size,
                 num_gcn_layer=3, parity='even', cutoff=5.0):
        super().__init__()
        assert parity in ['even', 'odd']
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.radial_embed_size = radial_embed_size
        self.radial_hidden_size = radial_hidden_size
        self.num_gcn_layer = num_gcn_layer
        self.parity = parity
        self.cutoff = cutoff

        self.embedding = nn.Embedding(100, num_radial)
        self.irreps_sh = o3.Irreps.spherical_harmonics(num_spherical, p=1 if parity == 'even' else -1)
        self.irreps_feat = (self.irreps_sh * num_radial).sort().irreps.simplify()
        self.gcns = nn.ModuleList([
            TFNLayer(
                (f'{num_radial}x0e' if i == 0 else self.irreps_feat),
                (f'{num_radial}x0e' if i == num_gcn_layer - 1 else self.irreps_feat),
                self.irreps_sh, radial_embed_size, radial_hidden_size
            ) for i in range(num_gcn_layer)
        ])
        self.act = NormActivation(self.irreps_feat)
        self.fc = nn.Sequential(
            nn.Linear(num_radial, 2 * num_radial),
            nn.SiLU(),
            nn.Linear(2 * num_radial, num_radial),
            nn.SiLU(),
            nn.Linear(num_radial, 1),
            nn.Sigmoid()
        )

    def forward(self, z, pos, batch):
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, loop=False)
        feat = self.embedding(z)

        src, dst = edge_index
        edge_vec = pos[src] - pos[dst]
        edge_len = edge_vec.norm(dim=-1) + 1e-8
        edge_feat = o3.spherical_harmonics(
            list(range(self.num_spherical + 1)), edge_vec / edge_len[..., None],
            normalize=False, normalization='integral'
        )
        edge_embed = soft_one_hot_linspace(
            edge_len, start=0.0, end=self.cutoff,
            number=self.radial_embed_size, basis='gaussian', cutoff=False
        ).mul(self.radial_embed_size ** 0.5)

        for i, gcn in enumerate(self.gcns):
            feat = gcn(edge_index, feat, edge_feat, edge_embed, dim_size=z.size(0))
            if i != self.num_gcn_layer - 1:
                feat = self.act(feat)
        out = scatter(feat, batch, dim=0)
        return self.fc(out).squeeze(-1)
