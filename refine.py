from argparse import Namespace
from time import time, strftime
from typing import Union
import torch
import sys
import torch.nn.functional as F
import os
import copy

from dataset import prepare_dataloaders
from models import JointModel, SeparateModel, SeparateOptimizer, CRF
from utils import set_seeds_all
from config import parser
from solve_proxy import solve_proxy
from train_loops import refine_spn, test_spn


def refine(args: Namespace, model_name: Union[None, str]=None, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Refine the SPN model by solving the maximin game in Eq. 3.

    Each iteration of refinement has two steps. In the first step, we run sum-product loopy belief propagation, which yields a collection of node and edge marginals as approximation to the marginals defined by p_theta. In the second step, we update the theta-functions parameterized by the node and edge GNNs to maximize Eq. 11.
    See train_loops.tune_gnn for more details.

    Args:
        args: Arguments from the command line. See config.py.
        model_name: The name of the model. Used in ckpt_dir and logging. If None, will be automatically generated.
        device: The device to be used.

    Returns:
        model: Trained SPN model.
    """
    # Log start time
    start = time()
    model_time = strftime('%m-%d_%H-%M-%S')

    # Get dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(args)
    
    # Initialize model according to args
    if args.GNN_model == 'CRF':
        model = CRF(args, train_loader.dataset.num_features, train_loader.dataset.num_classes).to(device)
    elif args.joint_model:
        model = JointModel(args, train_loader.dataset.num_features, train_loader.dataset.num_classes).to(device)
    else:
        model = SeparateModel(args, train_loader.dataset.num_features, train_loader.dataset.num_classes).to(device)

    # Train or load checkpoint
    if args.load_ckpt:
        model.load_state_dict(torch.load(args.load_ckpt))
    elif not args.no_proxy:
        model = solve_proxy(args, model=model, device=device)

    # Initialize optimizer according to args
    if isinstance(model, JointModel) or isinstance(model, CRF):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.refine_node_lr)
    elif isinstance(model, SeparateModel):
        steps = args.refine_epochs * len(train_loader.dataset)
        optimizer = SeparateOptimizer(args.refine_node_lr, steps, steps, model, torch.optim.Adam)
    else:
        raise NotImplementedError

    # Make ckpt_dir
    if model_name is None:
        if args.GNN_model == 'CRF':
            model_name = 'CRF'
        else:
            model_name = f"{'joint' if args.joint_model else 'separated'}_{args.GNN_model}{'' if args.no_proxy else '_refined'}{'' if not args.no_log_softmax else '_noLogSoftmax'}"
        model_name = f'{model_name}{args.log_str}'
    os.makedirs(os.path.join(args.ckpt_dir, model_name), exist_ok=True)

    # Initialize the best models and the best metrics
    best_model = copy.deepcopy(model.state_dict())
    best_val, best_test = (0.0, 0.0), (0.0, 0.0)

    # Refinement. Log to refine_log_file.
    with open(os.path.join(args.ckpt_dir, model_name, args.refine_log_file), 'a') as f:
        # Refinement
        for epoch in range(1, 1 + args.refine_epochs):
            loss_n, loss_e, loss_n_pos, loss_n_neg, loss_e_pos, loss_e_neg = \
                refine_spn(args, train_loader, model, optimizer, device)
            print(f'Epoch: {epoch:4d}, LossN: {loss_n:10.3g} ({loss_n_pos:10.3g} - {loss_n_neg:10.3g}), LossE: {loss_e:10.3g} ({loss_e_pos:10.3g} - {loss_e_neg:10.3g})', end='\r')
            if epoch % args.refine_eval_every == 0 or epoch == args.refine_epochs:
                node_val_acc, edge_val_acc = test_spn(args, val_loader, model, device)
                node_test_acc, edge_test_acc = test_spn(args, test_loader, model, device)
                print(f'Epoch: {epoch:4d}, NAcc (V, T): ({node_val_acc:.5f}, {node_test_acc:.5f}), EAcc (V, T): ({edge_val_acc:.5f}, {edge_test_acc:.5f})')
                # time,dataset,model,epoch,node_loss,edge_loss,n_val_acc,n_test_acc,e_val_acc,e_test_acc,run_time,seed
                f.write(f"{model_time},{args.dataset},{model_name},{epoch},{loss_n},{loss_e},{node_val_acc},{node_test_acc},{edge_val_acc},{edge_test_acc},{time() - start},{args.seed}\n")
                if node_val_acc > best_val[0]:
                    best_val, best_test = (node_val_acc, edge_val_acc), (node_test_acc, edge_test_acc)
                    best_model = copy.deepcopy(model.state_dict())

    # Refinement ends, restore best model
    model.load_state_dict(best_model)

    # Log SPN performance to rm_result_file (refinement result file)
    print(f'RM: NAcc (V, T): ({best_val[0]:.5f}, {best_test[0]:.5f}), EAcc (V, T): ({best_val[1]:.5f}, {best_test[1]:.5f})')
    with open(os.path.join(args.ckpt_dir, model_name, args.rm_result_file), 'a') as f:
        argv = ' '.join(sys.argv).replace('\n', '').replace(',', '_')
        # time,dataset,model,n_val_acc,n_test_acc,e_val_acc,e_test_acc,run_time,seed,argv,gamma
        f.write(f"{model_time},{args.dataset},{model_name},{best_val[0]},{best_test[0]},{best_val[1]},{best_test[1]},{time() - start},{args.seed},{argv},{args.edge_pred_softmax_temp}\n")

    # Save model
    model_save_path = os.path.join(args.ckpt_dir, model_name, f"model_{args.dataset}_seed{args.seed}_{model_time}_refine.pt")
    torch.save(best_model, model_save_path)
    print(f'Model saved to {model_save_path}.')

    # Save predictions (for PPI F1 score calculation)
    node_val_acc, edge_val_acc = test_spn(args, val_loader, model, device)
    node_test_acc, edge_test_acc, rm_test_pred = test_spn(args, test_loader, model, device, return_pred=True)
    rm_save_path = os.path.join(args.ckpt_dir, model_name, f"rm_{args.dataset}_seed{args.seed}_{model_time}.pt")
    torch.save(rm_test_pred, rm_save_path)

    return model


if __name__ == "__main__":
    torch.cuda.set_device(0)
    args = parser.parse_args()
    print(args)
    if args.seed:
        set_seeds_all(args.seed)

    refine(args)
