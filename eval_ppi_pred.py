"""
Evaluates models trained on PPI.
We first train SPN on 121 classification tasks separately and store the model predictions. Here we load the model predictions and calculate the overall accuracy and F1 scores.
"""


from argparse import ArgumentParser
from collections import defaultdict
import torch
import os
import pandas as pd
import numpy as np
from pathlib import Path
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_colwidth', 100)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    test_dataset = PPI(os.path.join('..', 'data', 'PPI'), split='test')
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    y_true = torch.cat([data.y for data in test_dataset], dim=0).to(device)
    y_true_np = y_true.detach().cpu().numpy()
    
    model_name_dict = dict(
        jointGAT='joint_GATConv',
        jointGCN='joint_GCNConv',
        jointGCN2='joint_GCN2Conv',
        jointSAGE='joint_SAGEConv',
        GAT='separated_GATConv',
        GCN='separated_GCNConv',
        GCN2='separated_GCN2Conv',
        SAGE='separated_SAGEConv',
        GraphUNet='separated_GraphUNet',
        CRF='CRF'
    )
    model_sources = dict(
        jointGAT=('pm', 'pp', 'crf', 'rm'),
        jointGCN=('pm', 'pp', 'crf', 'rm'),
        jointGCN2=('pm', 'pp', 'crf', 'rm'),
        jointSAGE=('pm', 'pp', 'crf', 'rm'),
        GAT=('pm', 'pp', 'crf', 'rm'),
        GCN=('pm', 'pp', 'crf', 'rm'),
        GCN2=('pm', 'pp', 'crf', 'rm'),
        SAGE=('pm', 'pp', 'crf', 'rm'),
        GraphUNet=('pm', 'pp', 'crf', 'rm'),
        CRF=('crf',)
    )
    p = ArgumentParser()
    p.add_argument('--ngraphs', type=int, nargs='+', default=[10])
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    p.add_argument('--log-str', type=str, default='')
    p.add_argument('--load-ckpt', type=str, default=os.path.join('.', 'results'))
    p.add_argument('--model-names', type=str, nargs='+', choices=('jointGAT', 'jointGCN', 'jointGCN2', 'jointSAGE', 'GAT', 'GCN2', 'SAGE', 'GraphUNet', 'GCN', 'CRF'), default=['GAT'])
    p.add_argument('--sources', type=str, nargs='+', default=('pm', 'pp', 'crf', 'rm'))
    p.add_argument('--pivot-save-to', type=str, default='pivot_ppi_result.csv')
    p.add_argument('--save-to', type=str, default='ppi_result.csv')
    args = p.parse_args()
    root = Path(args.load_ckpt)

    rows = []
    for model_name in args.model_names:
        for ngraphs in args.ngraphs:
            for seed in args.seeds:
                sources = model_sources[model_name]
                sources = [src for src in sources if src in args.sources]
                y_pred = {source: torch.zeros_like(y_true) for source in sources}
                valid_lids = defaultdict(list)
                for lid in range(121):
                    model_name_in_path = model_name_dict[model_name]
                    pm_ls = sorted(list(root.glob(f'{model_name_in_path}{args.log_str}/gnn_ppi-{ngraphs}-{lid}_seed{seed}_*.pt')))
                    pp_ls = sorted(list(root.glob(f'{model_name_in_path}{args.log_str}/pp_ppi-{ngraphs}-{lid}_seed{seed}_*.pt')))
                    rm_ls = sorted(list(root.glob(f'{model_name_in_path}_refined*{args.log_str}/rm_ppi-{ngraphs}-{lid}_seed{seed}_*.pt')))
                    crf_gnn_ls = sorted(list(root.glob(f'{model_name_in_path}_noLogSoftmax*{args.log_str}/rm_ppi-{ngraphs}-{lid}_seed{seed}_*.pt')))
                    crf_ls = sorted(list(root.glob(f'{model_name_in_path}{args.log_str}/rm_ppi-{ngraphs}-{lid}_seed{seed}_*.pt')))
                    if model_name == 'CRF':
                        pred_dict = {'crf': crf_ls}
                    else:
                        pred_dict = {'pm': pm_ls, 'pp': pp_ls, 'rm': rm_ls, 'crf': crf_gnn_ls}
                    for source in sources:
                        ls = pred_dict[source]
                        if len(ls) == 0:
                            print(f'Error: {model_name}\t{ngraphs}\t{lid}\t{seed}\t{source}')
                            continue
                        elif len(ls) > 1:
                            ls_repr = "\t".join(map(lambda x: x.name, ls))
                            print(f'Multi: {ls_repr}')
                        y_pred[source][:, lid] = torch.load(ls[0])
                        valid_lids[source].append(lid)
                for source in sources:
                    if len(valid_lids[source]) < 121:
                        missing_lids = set(range(121)).difference(valid_lids[source])
                        print(f'Error: {model_name}\t{ngraphs}\t{seed}\t{source}\t{len(missing_lids)}\t{",".join(map(str, missing_lids))}')
                        continue
                    pred = y_pred[source].detach().cpu().numpy()
                    f1 = f1_score(y_true_np, pred, average='micro')
                    acc = accuracy_score(y_true_np.flatten(), pred.flatten())
                    print(f"\tmodel: {model_name}, ngraphs: {ngraphs}, seed: {seed}, source: {source}, f1: {f1}, acc: {acc}")
                    rows.append((model_name, ngraphs, seed, source, f1, acc))
    df = pd.DataFrame(rows, columns=('model', 'ngraphs', 'seed', 'source', 'f1', 'acc'))
    df.to_csv(root / args.save_to)
    table = df.pivot_table(index=['model', 'ngraphs', 'source'], values=['f1', 'acc'], aggfunc=(np.mean, np.std))
    print(table)
    table.to_csv(root / args.pivot_save_to)
