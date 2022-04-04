from argparse import Namespace
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from torch_scatter import scatter_mean
from utils import soft_nll_loss
from belief_propagation import max_product_bp, sum_product_bp, get_potential


def solve_proxy_spn(args: Namespace, dataloader: DataLoader, model: nn.Module, optimizer: optim.Optimizer, device: torch.device):
    """
    Solve the proxy optimization problem of SPN for one epoch.

    Args:
        args: Arguments from the command line. See config.py.
        dataloader: DataLoader for the training set.
        model: The model to be trained.
        optimizer: Optimizer for training the model.
        device: The device to be used.

    Returns:
        total_node_loss: Node-level cross entropy for the training set.
        total_edge_loss: Edge-level cross entropy for the training set.
    """
    model.train()
    loss_op = nn.CrossEntropyLoss()

    total_node_loss, total_edge_loss = 0, 0
    for data in dataloader:
        data = data.to(device)

        optimizer.zero_grad()
        node_logit, edge_logit = model(data)

        node_loss = loss_op(node_logit, data.y)
        total_node_loss += node_loss.item() * data.num_graphs

        edge_loss = loss_op(edge_logit, data.edge_labels)
        total_edge_loss += edge_loss.item() * data.num_graphs

        loss = node_loss + args.solve_proxy_edge_lr/args.solve_proxy_node_lr * edge_loss

        loss.backward()
        optimizer.step()

    return total_node_loss, total_edge_loss


def refine_spn(args: Namespace, dataloader: DataLoader, model: nn.Module, optimizer: optim.Optimizer, device: torch.device):
    """
    Refine the SPN for one epoch.

    Args:
        args: Arguments from the command line. See config.py.
        dataloader: DataLoader for the training set.
        model: The model to be trained.
        optimizer: Optimizer for training the model.
        device: The device to be used.

    Returns:
        total_node_loss: Node-level loss for the training set.
        total_edge_loss: Edge-level loss for the training set.
        total_node_pos: Node-level positive loss for the training set. (See Eq. 11)
        total_node_neg: Node-level negative loss for the training set. (See Eq. 11)
        total_edge_pos: Edge-level positive loss for the training set. (See Eq. 11)
        total_edge_neg: Edge-level negative loss for the training set. (See Eq. 11)
    """
    model.train()
    total_node_loss, total_edge_loss, total_node_pos, total_node_neg, total_edge_pos, total_edge_neg = 0., 0., 0., 0., 0., 0.
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        pred_node, pred_edge = get_potential(args, data, model, dataloader.dataset.num_classes)

        node_marginal, edge_marginal = sum_product_bp(args, pred_node.detach(), pred_edge.detach(), data.edge_index, data.edge_index_reversed, device, dataloader.dataset.num_classes)

        node_target = data.y
        node_loss_pos = F.nll_loss(pred_node, node_target)
        node_loss_neg = soft_nll_loss(pred_node, node_marginal.detach())
        node_loss = node_loss_pos - node_loss_neg
        total_node_loss += node_loss.item()

        edge_target = data.edge_labels
        edge_loss_pos = F.nll_loss(pred_edge, edge_target)
        edge_loss_neg = soft_nll_loss(pred_edge, edge_marginal.detach())
        edge_loss = edge_loss_pos - edge_loss_neg
        total_edge_loss += edge_loss.item()
        
        total_node_pos += node_loss_pos.item()
        total_node_neg += node_loss_neg.item()
        total_edge_pos += edge_loss_pos.item()
        total_edge_neg += edge_loss_neg.item()

        loss = node_loss + args.refine_edge_lr/args.refine_node_lr * edge_loss
        loss.backward()
        optimizer.step()
    return total_node_loss, total_edge_loss, total_node_pos, total_node_neg, total_edge_pos, total_edge_neg


def _compute_accuracy(args: Namespace, pred: torch.Tensor, target: torch.Tensor, batch: torch.Tensor):
    correct = (pred == target).float()
    if args.dataset == 'dblp' or args.dataset.startswith('ppi'):
        acc = correct.mean().item()
    else:
        graph_acc = scatter_mean(correct, index=batch, dim=0, dim_size=batch.max().item() + 1)
        acc = (graph_acc == 1.).float().mean().item()
    return acc


@torch.no_grad()
def test_gnn(args: Namespace, dataloader: DataLoader, model: nn.Module, device: torch.device, return_pred: bool = False):
    """
    Predict node and edge labels separately with GNNs and calculate accuracies.

    Args:
        args: Arguments from the command line. See config.py.
        dataloader: DataLoader for the training set.
        model: The model to be trained.
        device: The device to be used.
        return_pred: Whether to return the node label predictions.

    Returns:
        node_acc: Node-level accuracy predicted by the node GNN.
        edge_acc: Edge-level accuracy predicted by the edge GNN.
        node_pred (optional): Predicted node labels. Only returned when return_pred is True.
    """
    model.eval()

    node_ys, node_preds, edge_ys, edge_preds = [], [], [], []
    batches, edge_batches = [], []
    for data in dataloader:
        data = data.to(device)
        batches.append(data.batch)
        edge_batches.append(data.batch[data.edge_index[0]])
        node_ys.append(data.y)
        edge_ys.append(data.edge_labels)

        node_logit, edge_logit = model(data)
        node_preds.append(torch.max(node_logit, dim=-1)[1])
        edge_preds.append(torch.max(edge_logit, dim=-1)[1])

    node_y, node_pred, edge_y, edge_pred, batch, edge_batch = map(lambda x: torch.cat(x, dim=0), (node_ys, node_preds, edge_ys, edge_preds, batches, edge_batches))
    node_acc = _compute_accuracy(args, node_pred, node_y, batch)
    edge_acc = _compute_accuracy(args, edge_pred, edge_y, edge_batch)

    if return_pred:
        return node_acc, edge_acc, node_pred
    else:
        return node_acc, edge_acc


@torch.no_grad()
def test_spn(args: Namespace, dataloader: DataLoader, model: nn.Module, device: torch.device, return_pred: bool = False):
    """
    Predict node and edge labels jointly with SPNs and calculate accuracies.

    Args:
        args: Arguments from the command line. See config.py.
        dataloader: DataLoader for the training set.
        model: The model to be trained.
        device: The device to be used.
        return_pred: Whether to return the node label predictions.

    Returns:
        node_acc: Node-level accuracy predicted by SPN.
        edge_acc: Edge-level accuracy predicted by SPN.
        node_pred (optional): Predicted node labels. Only returned when return_pred is True.
    """
    model.eval()
    y_ls, y_e_ls, p_s_ls, p_st_ls = [], [], [], []
    batches, edge_batches = [], []
    for data in dataloader:
        data = data.to(device)
        batches.append(data.batch)
        edge_batches.append(data.batch[data.edge_index[0]])
        pred_node, pred_edge = get_potential(args, data, model, dataloader.dataset.num_classes)
        p_s, p_st = max_product_bp(args, pred_node.detach(), pred_edge.detach(), data.edge_index, data.edge_index_reversed, device, dataloader.dataset.num_classes)
        y = data.y
        p_s_ls.append(p_s)
        p_st_ls.append(p_st)
        y_ls.append(y)
        y_e_ls.append(data.edge_labels)

    p_s = torch.cat(p_s_ls, dim=0)
    p_st = torch.cat(p_st_ls, dim=0)
    y = torch.cat(y_ls, dim=0)
    y_e = torch.cat(y_e_ls, dim=0)
    batch = torch.cat(batches, dim=0)
    edge_batch = torch.cat(edge_batches, dim=0)
    
    # calculate node predictions and accuracy
    _, y_pred = p_s.max(-1)
    acc_n = _compute_accuracy(args, y_pred, y, batch)

    # calculate edge predictions and accuracy
    _, y_pred_e = p_st.view(-1, dataloader.dataset.num_classes ** 2).max(-1)
    acc_e = _compute_accuracy(args, y_pred_e, y_e, edge_batch)

    if return_pred:
        return acc_n, acc_e, y_pred
    else:
        return acc_n, acc_e
