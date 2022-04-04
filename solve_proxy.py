from typing import Union
import torch
import copy
import os
import sys
from argparse import Namespace
from time import strftime, time

from models import JointModel, SeparateModel, SeparateOptimizer
from dataset import prepare_dataloaders
from config import parser
from utils import set_seeds_all
from train_loops import solve_proxy_spn, test_gnn, test_spn


def solve_proxy(args: Namespace, model: Union[JointModel, SeparateModel]=None,
             model_name: Union[None, str]=None,
             device: torch.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    """
    Train the SPN model with the proxy problem: optimize the node and edge GNNs to maximize the log-likelihood of the observed labels on nodes and edges, as shown in Eq. 10.

    Args:
        args: Arguments from the command line. See config.py.
        model: The model to be trained.
        model_name: The name of the model. Used in ckpt_dir and logging. If None, will be automatically generated.
        device: The device to be used.

    Returns:
        model: Trained SPN model.
    """

    # Log start time, make ckpt_dir
    start = time()
    model_time = strftime('%m-%d_%H-%M-%S')
    if model_name is None:
        model_name = 'joint' if args.joint_model else 'separated'
        model_name = f'{model_name}_{args.GNN_model}{args.log_str}'
    os.makedirs(os.path.join(args.ckpt_dir, model_name), exist_ok=True)

    # Get dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(args)

    # Initialize model and optimizer according to args
    if args.joint_model:
        if model is None:
            model = JointModel(args, train_loader.dataset.num_features, train_loader.dataset.num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.solve_proxy_node_lr)
    else:
        if model is None:
            model = SeparateModel(args, train_loader.dataset.num_features, train_loader.dataset.num_classes).to(device)
        steps = args.solve_proxy_epochs * len(train_loader.dataset)
        optimizer = SeparateOptimizer(args.solve_proxy_node_lr, steps, steps, model, torch.optim.Adam)
    print(model)

    # Proxy-solving. Log to proxy_log_file.
    with open(os.path.join(args.ckpt_dir, model_name, args.proxy_log_file), 'a') as f:
        # Test initial performance
        node_val_acc, edge_val_acc = test_gnn(args, val_loader, model, device)
        node_test_acc, edge_test_acc = test_gnn(args, test_loader, model, device)
        print(f'Epoch: {0:4d}, NAcc (V, T): ({node_val_acc:.5f}, {node_test_acc:.5f}), EAcc (V, T): ({edge_val_acc:.5f}, {edge_test_acc:.5f})')
        # time,dataset,model,epoch,node_loss,edge_loss,node_val_acc,node_test_acc,edge_val_acc,edge_test_acc,run_time,seed
        f.write(f"{model_time},{args.dataset},{model_name},{0},{0},{0},{node_val_acc},{node_test_acc},{edge_val_acc},{edge_test_acc},{time() - start},{args.seed}\n")

        # Initialize the best models and the best metrics
        best_node = best_edge = copy.deepcopy(model.state_dict())
        best_val, best_test = [0., 0.], [0., 0.]

        # Proxy-solving loop
        for epoch in range(1, 1 + args.solve_proxy_epochs):
            node_loss, edge_loss = solve_proxy_spn(args, train_loader, model, optimizer, device)
            print(f'Epoch: {epoch:4d}, Node Loss: {node_loss:.5f}, Edge Loss: {edge_loss:.5f}', end='\r')
            if epoch % args.solve_proxy_eval_every == 0 or epoch == args.solve_proxy_epochs:
                node_val_acc, edge_val_acc = test_gnn(args, val_loader, model, device)
                node_test_acc, edge_test_acc = test_gnn(args, test_loader, model, device)
                print(f'Epoch: {epoch:4d}, NAcc (V, T): ({node_val_acc:.5f}, {node_test_acc:.5f}), EAcc (V, T): ({edge_val_acc:.5f}, {edge_test_acc:.5f})')
                # time,dataset,model,epoch,node_loss,edge_loss,node_val_acc,node_test_acc,edge_val_acc,edge_test_acc,run_time,seed
                f.write(f"{model_time},{args.dataset},{model_name},{epoch},{node_loss},{edge_loss},{node_val_acc},{node_test_acc},{edge_val_acc},{edge_test_acc},{time() - start},{args.seed}\n")
                if args.joint_model and not args.separate_model_after_proxy:
                    if node_val_acc ** 2 + edge_val_acc > best_val[0] ** 2 + best_val[1]:
                        best_val, best_test = [node_val_acc, edge_val_acc], [node_test_acc, edge_test_acc]
                        best_node = best_edge = copy.deepcopy(model.state_dict())
                else:
                    if node_val_acc > best_val[0]:
                        best_val[0], best_test[0] = node_val_acc, node_test_acc
                        best_node = copy.deepcopy(model.state_dict())
                    if edge_val_acc > best_val[1]:
                        best_val[1], best_test[1] = edge_val_acc, edge_test_acc
                        best_edge = copy.deepcopy(model.state_dict())

    # Proxy-solving ends, restore best model
    model = model.combine_weights(best_node, best_edge, device)

    # Log GNN performance to pm_result_file (pseudomarginal result file)
    print(f'PM: NAcc (V, T): ({best_val[0]:.5f}, {best_test[0]:.5f}), EAcc (V, T): ({best_val[1]:.5f}, {best_test[1]:.5f})')
    argv = ' '.join(sys.argv).replace('\n', '').replace(',', '_')
    with open(os.path.join(args.ckpt_dir, model_name, args.pm_result_file), 'a') as f:
        # time,dataset,model,node_val_acc,node_test_acc,edge_val_acc,edge_test_acc,run_time,seed,argv,gamma
        f.write(f"{model_time},{args.dataset},{model_name},{best_val[0]},{best_test[0]},{best_val[1]},{best_test[1]},{time() - start},{args.seed},{argv},{0}\n")
    _, _, gnn_test_pred = test_gnn(args, test_loader, model, device, return_pred=True)

    # Select best edge_pred_softmax_temp from candidates
    best_val, best_test, best_bp_test_pred, best_edge_pred_softmax_temp = [0., 0.], [0., 0.], None, 0.
    for edge_pred_softmax_temp in args.edge_pred_softmax_temp_candidates:
        args.edge_pred_softmax_temp = edge_pred_softmax_temp
        node_val_acc, edge_val_acc = test_spn(args, val_loader, model, device)
        node_test_acc, edge_test_acc, bp_test_pred = test_spn(args, test_loader, model, device, return_pred=True)
        print(f'PP: NAcc (V, T): ({node_val_acc:.5f}, {node_test_acc:.5f}), EAcc (V, T): ({edge_val_acc:.5f}, {edge_test_acc:.5f}), edge_pred_softmax_temp: {edge_pred_softmax_temp}')
        if node_val_acc > best_val[0]:
            best_val, best_test, best_bp_test_pred = [node_val_acc, edge_val_acc], [node_test_acc, edge_test_acc], bp_test_pred
            best_edge_pred_softmax_temp = edge_pred_softmax_temp
    args.edge_pred_softmax_temp = best_edge_pred_softmax_temp
    print(f'Choosing {best_edge_pred_softmax_temp} as edge_pred_softmax_temp from {args.edge_pred_softmax_temp_candidates}.')

    # Log SPN performance to pp_result_file (pseudomarginal result file)
    with open(os.path.join(args.ckpt_dir, model_name, args.pp_result_file), 'a') as f:
        # time,dataset,model,n_val_acc,n_test_acc,e_val_acc,e_test_acc,run_time,seed,argv,gamma
        f.write(f"{model_time},{args.dataset},{model_name},{best_val[0]},{best_test[0]},{best_val[1]},{best_test[1]},{0},{args.seed},{argv},{args.edge_pred_softmax_temp}\n")

    # Save model
    model_save_path = os.path.join(args.ckpt_dir, model_name, f"model_{args.dataset}_seed{args.seed}_{model_time}_proxy.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}.')

    # Save predictions (for PPI F1 score calculation)
    gnn_pred_save_path = os.path.join(args.ckpt_dir, model_name, f"gnn_{args.dataset}_seed{args.seed}_{model_time}.pt")
    torch.save(gnn_test_pred, gnn_pred_save_path)
    bp_pred_save_path = os.path.join(args.ckpt_dir, model_name, f"pp_{args.dataset}_seed{args.seed}_{model_time}.pt")
    torch.save(best_bp_test_pred, bp_pred_save_path)

    return model


if __name__ == "__main__":
    torch.cuda.set_device(0)
    args = parser.parse_args()
    print(args)
    if args.seed:
        set_seeds_all(args.seed)
    
    solve_proxy(args)
