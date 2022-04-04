from argparse import Namespace
from typing import Tuple
import torch
from torch import nn
from torch_geometric.data import Data


def belief_propagation(sum_product: bool, args: Namespace, pred_node: torch.Tensor, pred_edge: torch.Tensor, edge_index: torch.Tensor, reverse_mapping: torch.Tensor, device: torch.device, num_classes: int):
    """
    Perform sum/max-product loopy belief propagation. (Eq. 12)

    Args:
        sum_product: If true, compute sum-product; else, compute max-product.
        args: Arguments from the command line. See config.py.
        pred_node: node potential computed by `get_potential`.
        pred_edge: edge potential computed by `get_potential`.
        edge_index: edge index tensor of the (batch of) graph.
        reverse_mapping: Gives the index of the reverse edge in edge_index. Suppose edge_index[:, i] = (s, t), then edge_index[reverse_mapping[i]] = (t, s).
        device: device to perform the computation.
        num_classes: number of label classes in the dataset

    Returns:
        q_s: inferred node marginal.
        q_st: inferred edge marginal.
    """
    n_nodes, n_edges = pred_node.size(0), pred_edge.size(0)
    pred_edge = pred_edge.view(-1, num_classes, num_classes)

    msg = torch.ones(n_edges, num_classes).cuda() / num_classes
    msg = msg.log()

    s, t = edge_index[0], edge_index[1]
    for _ in range(args.sum_prod_bp_steps if sum_product else args.max_prod_bp_steps):
        # perform message passing
        # M_{s -> t} (x_t) = 1/Z \sum_{x_s^'} \exp( \theta_{st} (x_s^', x_t) + \theta_s (x_s) + \sum_{u \in N(s)\\t} \log M_{u -> s}(x_s^') )
        vec = pred_edge.clone()  # (n_edges, num_classes, num_classes)
        vec += pred_node[s].unsqueeze(2)  # (n_edges, num_classes, num_classes) + (n_edges, num_classes, 1)
        msg_us = torch.zeros((n_nodes, num_classes), dtype=msg.dtype, device=device).scatter_add_(
            dim=0,
            index=t.unsqueeze(1).expand(-1, num_classes),
            src=msg
        )  # (n_nodes, num_classes)
        msg_us = msg_us[s] * args.edge_appearance_prob - msg[reverse_mapping]  # (n_edges, num_classes)
        vec += msg_us.unsqueeze(2)  # (n_edges, num_classes, num_classes) + (n_edges, num_classes, 1)
        if sum_product:
            vec = vec.sum(1)  # (n_edges, num_classes, num_classes) -> (n_edges, num_classes)
        else:
            vec, _ = vec.max(1)  # (n_edges, num_classes, num_classes) -> (n_edges, num_classes)
        msg = vec.log_softmax(-1)

    # calculate the logit of node marginals
    q_s = pred_node.clone()  # (n_nodes, num_classes)
    q_s += args.edge_appearance_prob * torch.zeros((n_nodes, num_classes), dtype=msg.dtype, device=device).scatter_add_(
        dim=0,
        index=t.unsqueeze(1).expand(-1, num_classes),
        src=msg
    )  # (n_nodes, num_classes)
    
    # calculate the logit of edge marginals
    q_st = pred_edge.clone()  # (n_edges, num_classes, num_classes)
    q_st += q_s[s].unsqueeze(-1)  # \theta_s(x_s): (n_nodes, num_classes) -> (n_edges, num_classes) -> (n_edges, num_classes, 1)
    q_st += q_s[t].unsqueeze(-2)  # \theta_t(x_t): (n_nodes, num_classes) -> (n_edges, num_classes) -> (n_edges, 1, num_classes)
    q_st -= msg.unsqueeze(-2)  # M_{st}(x_t): (n_edges, num_classes) -> (n_edges, 1, num_classes)
    q_st -= msg[reverse_mapping].unsqueeze(-1)  # M_{ts}(x_s): (n_edges, num_classes) -> (n_edges, num_classes, 1)

    q_s = torch.softmax(args.marginal_softmax_temp * q_s, dim=-1).detach()
    q_st = torch.softmax(args.marginal_softmax_temp * q_st.view(-1, num_classes ** 2), dim=-1).detach()

    return q_s, q_st


def max_product_bp(args: Namespace, pred_node: torch.Tensor, pred_edge: torch.Tensor, edge_index: torch.Tensor, reverse_mapping: torch.Tensor, device: torch.device, num_classes: int):
    with torch.no_grad():
        return belief_propagation(sum_product=False, args=args, pred_node=pred_node, pred_edge=pred_edge, edge_index=edge_index, reverse_mapping=reverse_mapping, device=device, num_classes=num_classes)


def sum_product_bp(args: Namespace, pred_node: torch.Tensor, pred_edge: torch.Tensor, edge_index: torch.Tensor, reverse_mapping: torch.Tensor, device: torch.device, num_classes: int):
    with torch.no_grad():
        return belief_propagation(sum_product=True, args=args, pred_node=pred_node, pred_edge=pred_edge, edge_index=edge_index, reverse_mapping=reverse_mapping, device=device, num_classes=num_classes)


def get_potential(args: Namespace, data: Data, model: nn.Module, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Computes node and edge potentials (theta-functions) using Eq. 9.
    The potentials define the joint label distribution in Eq. 2.

    Args:
        args: Arguments from the command line. See config.py.
        data: (batch of) data for computing the potentials
        model: GNN model that outputs node and edge logits
        num_classes: number of label classes in the dataset

    Returns:
        pred_node: predicted node potential \theta_{s} of size [batch_size, num_classes]
        pred_edge: predicted edge potential \theta_{st} of size [batch_size, num_classes ** 2]
    """
    node_logit, edge_logit = model(data)
    if args.no_log_softmax:
        return node_logit, edge_logit
    pred_node = torch.log_softmax(node_logit, dim=-1)
    eps = torch.finfo(edge_logit.dtype).eps
    logits = torch.softmax(args.edge_pred_softmax_temp * edge_logit, dim=-1).view(-1, num_classes, num_classes) + eps
    sum_s = torch.sum(logits, dim=2).unsqueeze(2) + eps
    sum_t = torch.sum(logits, dim=1).unsqueeze(1) + eps
    pred_edge = (logits.log() - args.edge_marginal_norm_coef * (sum_s.log() + sum_t.log())).view(-1, num_classes ** 2)
    return pred_node, pred_edge
