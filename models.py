from math import e
from typing import Iterable, OrderedDict, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GCN2Conv, GINConv, GENConv, GraphUNet
from argparse import Namespace


class norm_act_drop(torch.nn.Module):
    def __init__(self, size: int, norm_module: str, activation: str, dropout_prob: float, final_layer: bool = False):
        super().__init__()
        self.norm = self.get_norm_layer(size, norm_module) if norm_module != 'none' else None
        self.activation, self.dropout = None, None
        if not final_layer:
            self.activation = getattr(torch.nn, activation)()
            self.dropout = torch.nn.Dropout(dropout_prob) if dropout_prob else None

    @staticmethod
    def get_norm_layer(size, norm_module='none'):
        if norm_module == 'none':
            return torch.nn.Identity()
        elif norm_module == 'batch':
            return torch.nn.BatchNorm1d(size)
        elif norm_module == 'instance':
            return torch.nn.InstanceNorm1d(size)
        elif norm_module == 'layer':
            return torch.nn.LayerNorm(size)
        else:
            return NotImplementedError(f"Not Implemented norm layer {norm_module}")

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_sizes, activation='ReLU', norm_module='none', dropout_prob=0.3):
        super().__init__()
        mlp_layers = []
        for hidden_size in hidden_sizes:
            mlp_layers.append(torch.nn.Linear(in_features, hidden_size))
            mlp_layers.append(norm_act_drop(hidden_size, norm_module, activation, dropout_prob))
            in_features = hidden_size
        if out_features:
            mlp_layers.append(torch.nn.Linear(in_features, out_features))
        self.mlp = torch.nn.Sequential(*mlp_layers)
    def forward(self, x):
        return self.mlp(x)


class GNNLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int, args: Namespace, final_layer: bool, layer: int):
        super().__init__()
        self.args = args
        self.conv = self.get_conv(in_channels, out_channels, heads, args, final_layer, layer)

        if args.skip_connections == 'linear':
            self.lin = torch.nn.Linear(in_channels, out_channels)
        elif args.skip_connections == 'identity':
            assert in_channels == out_channels, f'Identity skip connection requires in_channels == out_channels, received in_channels {in_channels} and out_channels {out_channels}.'
        elif args.skip_connections not in ('none',):
            raise NotImplementedError(f'Not implemented skip connection function: {args.skip_connections}.')

        self.norm_act_drop_before_conv = hasattr(args, 'norm_act_drop_before_conv') and args.norm_act_drop_before_conv
        self.norm_act_drop = norm_act_drop(out_channels, norm_module=args.GNN_norm_module, activation=args.GNN_activation, dropout_prob=args.dropout_prob, final_layer=final_layer)

    def get_conv(self, in_channels: int, out_channels: int, heads: int, args: Namespace, final_layer: bool, layer: int):
        if args.GNN_model == 'GATConv':
            if not final_layer:
                assert out_channels % heads == 0
                return GATConv(in_channels, out_channels // heads, heads, concat=True)
            else:
                return GATConv(in_channels, out_channels, heads, concat=False)
        elif args.GNN_model == 'GINConv':
            return GINConv(nn=MLP(in_channels, out_channels, args.GIN_hidden_sizes, args.GIN_activation, args.MLP_norm_module, args.dropout_prob))
        elif args.GNN_model == 'GCNConv':
            return GCNConv(in_channels, out_channels, improved=args.GCN_improved, normalize=args.GNN_normalize)
        elif args.GNN_model == 'SAGEConv':
            return SAGEConv(in_channels, out_channels, normalize=args.GNN_normalize)
        elif args.GNN_model == 'GCN2Conv':
            assert in_channels == out_channels, f'{in_channels} != {out_channels}'
            return GCN2Conv(in_channels, alpha=args.GCN2_alpha, theta=args.GCN2_theta, layer=layer, shared_weights=args.GCN2_shared_weights, normalize=args.GNN_normalize)
        elif args.GNN_model == 'GENConv':
            return GENConv(in_channels, out_channels, aggr='softmax', learn_t=args.GEN_learn_temp, num_layers=2, norm=args.MLP_norm_module)
        else:
            raise NotImplementedError('GNN model not implemented')

    def forward(self, x, *args, **kwargs):
        x0 = x
        if self.norm_act_drop is not None and self.norm_act_drop_before_conv:
            x = self.norm_act_drop(x)

        if self.conv is not None and self.args.ckpt_grad and x.requires_grad:
            z = checkpoint(self.conv, x, *args, **kwargs)
        else:
            z = self.conv(x, *args, **kwargs)

        if self.args.skip_connections == 'linear':
            x = (self.lin(x) + z)
        elif self.args.skip_connections == 'identity':
            x = x0 + z
        else:
            x = z

        if self.norm_act_drop is not None and not self.norm_act_drop_before_conv:
            x = self.norm_act_drop(x)
        return x


class MultiGNNLayers(torch.nn.Module):
    def __init__(self, args: Namespace, num_features: int):
        super().__init__()
        self.args = args
        self.norm_act_drop_before_conv = hasattr(args, 'norm_act_drop_before_conv') and args.norm_act_drop_before_conv
        if args.GNN_model == 'GATConv':
            assert len(args.GNN_hidden_sizes) == len(args.GAT_heads) - (0 if args.skip_connections == 'identity' else 1)
        self.pre_layer = None
        self.layers = torch.nn.ModuleList()
        in_channels = num_features
        if args.skip_connections == 'identity' or args.GNN_model == 'GCN2Conv':
            hidden_size = args.GNN_hidden_sizes[0]
            assert (np.array(args.GNN_hidden_sizes) == hidden_size).all()
            if self.norm_act_drop_before_conv:
                self.pre_layer = MLP(in_channels, hidden_size, tuple(), activation=args.MLP_activation, norm_module=args.MLP_norm_module, dropout_prob=args.dropout_prob)
            else:
                self.pre_layer = MLP(in_channels, 0, (hidden_size,), activation=args.MLP_activation, norm_module=args.MLP_norm_module, dropout_prob=args.dropout_prob)
            in_channels = hidden_size
        for i, size in enumerate(args.GNN_hidden_sizes):
            heads = args.GAT_heads[i] if args.GNN_model == 'GATConv' else None
            self.layers.append(GNNLayer(in_channels, size, heads, args=args, final_layer=False, layer=i + 1))
            in_channels = size

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.pre_layer is not None:
            x = self.pre_layer(x)
        if self.norm_act_drop_before_conv:
            x = self.layers[0].conv(x, edge_index)
            for layer in self.layers[1:]:
                x = layer(x, edge_index)
            x = self.layers[0].norm_act_drop(x)
        else:
            if self.args.GNN_model == 'GCN2Conv':
                x0 = x
            for layer in self.layers:
                if self.args.GNN_model == 'GCN2Conv':
                    x = layer(x, x0, edge_index)
                else:
                    x = layer(x, edge_index)
        return x


class GraphUNetWithDropout(torch.nn.Module):
    def __init__(self, args: Namespace, num_features: int):
        super().__init__()
        self.dropout_prob = args.dropout_prob
        pool_ratios = 0.5
        hidden_size = args.GNN_hidden_sizes[0]
        assert (np.array(args.GNN_hidden_sizes) == hidden_size).all()
        self.unet = GraphUNet(num_features, hidden_size, hidden_size,
                              depth=len(args.GNN_hidden_sizes), pool_ratios=pool_ratios)

    def forward(self, data):
        edge_index, _ = dropout_adj(data.edge_index, p=0.2,
                                    force_undirected=True,
                                    num_nodes=data.num_nodes,
                                    training=self.training)
        x = data.x
        if self.dropout_prob:
            x = F.dropout(x, p=self.dropout_prob, training=self.training)

        x = self.unet(x, edge_index)
        return x


class NodeClassifier(torch.nn.Module):
    def __init__(self, args: Namespace, in_channels: int, num_classes: int = 2):
        super().__init__()
        self.args = args
        if args.skip_connections == 'identity':
            self.final_layer = torch.nn.Linear(in_channels, num_classes)
        else:
            heads = args.GAT_heads[-1] if args.GNN_model == 'GATConv' else None
            self.final_layer = GNNLayer(in_channels, num_classes, heads, args=args, final_layer=True, layer=len(args.GNN_hidden_sizes))
    def forward(self, node_repr, data):
        if self.args.skip_connections == 'identity':
            node_pred = self.final_layer(node_repr)
        else:
            node_pred = self.final_layer(node_repr, data.edge_index)
        return node_pred


class NodeGNN(torch.nn.Module):
    def __init__(self, args: Namespace, num_features: int, num_classes: int = 2):
        super().__init__()
        self.args = args
        model = GraphUNetWithDropout if args.GNN_model == 'GraphUNet' else MultiGNNLayers
        self.multi_gnn_layers = model(args, num_features)
        self.node_clf = NodeClassifier(args, args.GNN_hidden_sizes[-1], num_classes)

    def forward(self, data):
        node_repr = self.multi_gnn_layers(data)
        return self.node_clf(node_repr, data)


class EdgeClassifier(torch.nn.Module):
    def __init__(self, args: Namespace, in_channels: int, num_classes: int = 2):
        super().__init__()
        self.args = args
        if args.skip_connections == 'identity' or args.GNN_model == 'GCN2Conv':
            args.edge_hidden_sizes = list(args.edge_hidden_sizes)
            args.edge_hidden_sizes[0] = in_channels
        else:
            heads = args.GAT_heads[-1] if args.GNN_model == 'GATConv' else None
            self.final_layer = GNNLayer(in_channels, args.edge_hidden_sizes[0], heads, args=args, final_layer=True, layer=len(args.GNN_hidden_sizes))

        args.edge_hidden_sizes = np.array(args.edge_hidden_sizes, dtype=int)
        if args.split_repr:
            args.edge_hidden_sizes = args.edge_hidden_sizes // 2
        if args.edge_label_predictor == 'concat':
            self.edge_label_predictor = MLP(args.edge_hidden_sizes[0] * 2, num_classes ** 2, args.edge_hidden_sizes[1:],
                args.MLP_activation, args.MLP_norm_module, args.dropout_prob)
        elif args.edge_label_predictor == 'bilinear':
            if len(args.edge_hidden_sizes) > 1:
                self.edge_label_predictor = torch.nn.ModuleList([
                    torch.nn.Bilinear(args.edge_hidden_sizes[0], args.edge_hidden_sizes[0], args.edge_hidden_sizes[1]),
                    MLP(args.edge_hidden_sizes[1], num_classes ** 2, args.edge_hidden_sizes[2:],
                        args.MLP_activation, args.MLP_norm_module, args.dropout_prob)
                ])
            else:
                self.edge_label_predictor = torch.nn.Bilinear(args.edge_hidden_sizes[0], args.edge_hidden_sizes[0], num_classes ** 2)
        else:
            raise NotImplementedError('Edge label predictor not implemented.')

    def forward(self, edge_repr, data):
        if self.args.skip_connections != 'identity' and self.args.GNN_model != 'GCN2Conv':
            edge_repr = self.final_layer(edge_repr, data.edge_index)
        
        if self.args.split_repr:
            out_repr, in_repr = torch.split(edge_repr, edge_repr.size(1) // 2, dim=1)
            u, v = out_repr[data.edge_index[0]], in_repr[data.edge_index[1]]
        else:
            u, v = edge_repr[data.edge_index[0]], edge_repr[data.edge_index[1]]
        if self.args.edge_label_predictor == 'concat':
            uv = torch.cat([u, v], dim=-1)
            edge_pred = self.edge_label_predictor(uv)
        elif self.args.edge_label_predictor == 'bilinear':
            if isinstance(self.edge_label_predictor, torch.nn.Bilinear):
                edge_pred = self.edge_label_predictor(u, v)
            else:
                edge_pred = self.edge_label_predictor[0](u, v)
                edge_pred = self.edge_label_predictor[1](edge_pred)
        else:
            raise NotImplementedError('Edge label predictor not implemented.')
        return edge_pred


class EdgeGNN(torch.nn.Module):
    def __init__(self, args: Namespace, num_features: int, num_classes: int = 2):
        super().__init__()
        self.args = args
        model = GraphUNetWithDropout if args.GNN_model == 'GraphUNet' else MultiGNNLayers
        self.multi_gnn_layers = model(args, num_features)
        dropout_prob = max(args.edge_repr_dropout, args.dropout_prob)
        self.dropout = torch.nn.Dropout(args.edge_repr_dropout) if dropout_prob else None
        self.edge_clf = EdgeClassifier(args, args.GNN_hidden_sizes[-1], num_classes)

    def forward(self, data):
        node_repr = self.multi_gnn_layers(data)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        return self.edge_clf(node_repr, data)


class JointModel(torch.nn.Module):
    def __init__(self, args: Namespace, num_features: int, num_classes: int = 2):
        super().__init__()
        self.args, self.num_features, self.num_classes = args, num_features, num_classes
        if args.GNN_model == 'GATConv':
            assert len(args.GNN_hidden_sizes) == len(args.GAT_heads) - 1
        model = GraphUNetWithDropout if args.GNN_model == 'GraphUNet' else MultiGNNLayers
        self.multi_gnn_layers = model(args, num_features)
        in_channels = args.GNN_hidden_sizes[-1]
        self.node_clf = NodeClassifier(args, in_channels, num_classes)
        self.edge_clf = EdgeClassifier(args, in_channels, num_classes)
        dropout_prob = max(args.edge_repr_dropout, args.dropout_prob)
        self.dropout = torch.nn.Dropout(args.edge_repr_dropout) if dropout_prob else None

    def forward(self, data):
        node_repr = self.multi_gnn_layers(data)
        node_pred = self.node_clf(node_repr, data)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        edge_pred = self.edge_clf(node_repr, data)
        return node_pred, edge_pred

    def combine_weights(self, node_weights: OrderedDict, edge_weights: OrderedDict, device: torch.device):
        if self.args.separate_model_after_pretraining:
            new_model = SeparateModel(self.args, self.num_features, self.num_classes).to(device)
            node_weights = OrderedDict(
                [(f'node_gnn.{name}', w) for name, w in node_weights.items() if name.startswith('multi_gnn_layers') or name.startswith('node_clf')]
            )
            edge_weights = OrderedDict(
                [(f'edge_gnn.{name}', w) for name, w in edge_weights.items() if name.startswith('multi_gnn_layers') or name.startswith('edge_clf')]

            )
            return new_model.combine_weights(node_weights, edge_weights, device)
        else:
            self.load_state_dict(edge_weights)
            return self
    

class SeparateModel(torch.nn.Module):
    def __init__(self, args: Union[Namespace, Iterable[Namespace]], num_features: int, num_classes: int = 2):
        super().__init__()
        if isinstance(args, Namespace):
            node_args = edge_args = args
        else:
            node_args, edge_args = args
        self.node_gnn = NodeGNN(node_args, num_features, num_classes)
        self.edge_gnn = EdgeGNN(edge_args, num_features, num_classes)

    def forward(self, data):
        return self.node_gnn(data), self.edge_gnn(data)

    def combine_weights(self, node_weights: OrderedDict, edge_weights: OrderedDict, device: torch.device):
        self.to(device)
        node_weights = OrderedDict(
            [(name[9:], w) for name, w in node_weights.items() if name.startswith('node_gnn')]
        )
        edge_weights = OrderedDict(
            [(name[9:], w) for name, w in edge_weights.items() if name.startswith('edge_gnn')]
        )
        self.node_gnn.load_state_dict(node_weights)
        self.edge_gnn.load_state_dict(edge_weights)
        return self


class SeparateOptimizer:
    def __init__(self, lr, node_epochs, edge_epochs, separable_model: SeparateModel, cls=torch.optim.Adam):
        node_optim = cls(separable_model.node_gnn.parameters(), lr=lr)
        edge_optim = cls(separable_model.edge_gnn.parameters(), lr=lr)  # We will have a args.pretrain_edge_lr/args.pretrain_node_lr coef when calculating the edge loss
        self.optims = (node_optim, edge_optim)
        self.remaining_steps = [node_epochs, edge_epochs]
    
    def zero_grad(self):
        for optim in self.optims:
            optim.zero_grad()
    
    def step(self):
        for i, (optim, rem_step) in enumerate(zip(self.optims, self.remaining_steps)):
            if rem_step > 0:
                optim.step()
                self.remaining_steps[i] -= 1

    def state_dict(self):
        return self.optims[0].state_dict(), self.optims[1].state_dict()


class CRF(torch.nn.Module):
    def __init__(self, args: Namespace, num_features: int, num_classes: int = 2):
        super().__init__()
        self.args = args
        self.node_potential = torch.nn.Linear(num_features, num_classes)
        if args.edge_label_predictor == 'concat':
            self.edge_potential = torch.nn.Linear(num_features * 2, num_classes ** 2)
        elif args.edge_label_predictor == 'bilinear':
            self.edge_potential = torch.nn.Bilinear(num_features, num_features, num_classes ** 2)
        else:
            raise NotImplementedError('Edge label predictor not implemented.')

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        u, v = x[edge_index[0]], x[edge_index[1]]
        if self.args.edge_label_predictor == 'concat':
            uv = torch.cat([u, v], dim=-1)
            edge_potential = self.edge_potential(uv)
        elif self.args.edge_label_predictor == 'bilinear':
            edge_potential = self.edge_potential(u, v)
        else:
            raise NotImplementedError('Edge label predictor not implemented.')
        return self.node_potential(x), edge_potential

