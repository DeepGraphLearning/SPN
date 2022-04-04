import argparse
import os

activations = ("Threshold", "ReLU", "Hardtanh", "ReLU6", "Sigmoid", "Tanh", "Softmax", "Softmax2d",
    "LogSoftmax", "ELU", "SELU", "CELU", "GELU", "Hardshrink", "LeakyReLU", "LogSigmoid", "Softplus", "Softshrink",
    "MultiheadAttention", "PReLU", "Softsign", "Softmin", "Tanhshrink", "RReLU", "GLU", "Hardsigmoid")

parser = argparse.ArgumentParser()

# dataset params
parser.add_argument('--dataset', type=str, default='ppi-1-0', help='name of the dataset, should be one of {ppi-[n_graphs]-[lid], cora, citeseer, pubmed, dblp}')

# general training params
parser.add_argument('--seed', type=int, default=1, help='Random seed for all random number generator in the program. Set to 0 to disable setting seed.')

# proxy-solving params
parser.add_argument('--no-proxy', action='store_true', help='do not solve proxy in the model')
parser.add_argument('--solve-proxy-epochs', type=int, default=400, help='Number of epochs to train the JointModel.')
parser.add_argument('--solve-proxy-node-lr', type=float, default=5e-3, help='learning rate for the node GNN.')
parser.add_argument('--solve-proxy-edge-lr', type=float, default=1e-3, help='learning rate for the edge GNN.')
parser.add_argument('--solve-proxy-eval-every', type=int, default=25, help='Evaluate model every eval_every epochs during proxy solving.')
parser.add_argument('--separate-model-after-proxy', action='store_true', help='If True, will convert a JointModel into a SeparateModel after proxy solving.')

# refinement params
parser.add_argument('--refine-epochs', type=int, default=300, help='Number of epochs to refine the GNN(s) with BP + GD.')
parser.add_argument('--refine-node-lr', type=float, default=1e-5, help='learning rate for the node GNN.')
parser.add_argument('--refine-edge-lr', type=float, default=1e-5, help='learning rate for the edge GNN.')
parser.add_argument('--refine-eval-every', type=int, default=5, help='Evaluate model every refine_eval_every epochs during fine tuning.')

# save & restore params
parser.add_argument('--ckpt-dir', type=str, default=os.path.join('.', 'results'), help='directory for model saving and restoring')
parser.add_argument('--load-ckpt', type=str, default='', help='model checkpoint file')

# logging paramss
parser.add_argument('--log-str', type=str, default='', help='additional string to log to results.csv')
parser.add_argument('--proxy-log-file', type=str, default='proxy_log.csv', help='file name (under args.ckpt_dir) for writing proxy-solving logs')
parser.add_argument('--pm-result-file', type=str, default='pm_result.csv', help='file name (under args.ckpt_dir) for writing results of PseudoMarginals (PM)')
parser.add_argument('--pp-result-file', type=str, default='pp_result.csv', help='file name (under args.ckpt_dir) for writing results of Proxy Problem (PP)')
parser.add_argument('--refine-log-file', type=str, default='refine_log.csv', help='file name (under args.ckpt_dir) for writing fine tuning logs')
parser.add_argument('--rm-result-file', type=str, default='refine_result.csv', help='file name (under args.ckpt_dir) for writing results of RefineMent (RM)')

# model params
parser.add_argument('-j', '--joint-model', action='store_true', help='If True, train a JointModel model; else, separately train a SeparateModel where node and edge GNNs do not share parameters.')
parser.add_argument('--no-log-softmax', action='store_true', help='If True, get_potential will output the model prediction as potential directly.')
parser.add_argument('--split-repr', action='store_true', help='If True, will split representation for out-node and in-node in edge gnn.')
parser.add_argument('--MLP-activation', type=str, choices=activations, default='ReLU', help='Activation module to use for the MLP layers.')
parser.add_argument('--MLP-norm-module', type=str, choices=('none', 'batch', 'layer', 'instance'), default='none', help='Type of normalization module in MLP.')
parser.add_argument('--edge-repr-dropout', type=float, default=0.5, help='input dropout probability for EdgeClassifier.')
parser.add_argument('--edge-label-predictor', type=str, choices=('concat', 'bilinear'), default='concat', help='Model to predict the edge labels.')
parser.add_argument('--edge-hidden-sizes', type=int, nargs='+', default=(64,), help='hidden layer sizes of the predictor MLP of the edge classifier')
parser.add_argument('-d', '--dropout-prob', type=float, default=0, help='dropout probabilities, 0 for no dropout.')
parser.add_argument('--GNN-norm-module', type=str, choices=('none', 'batch', 'layer', 'instance'), default='none', help='Type of normalization module in GNN.')
parser.add_argument('--ckpt-grad', action='store_true', help='Enables module checkpointing, which costs less memory but more run time.')

# GNN specific params
subparsers = parser.add_subparsers()
# Best params for GAT:
# --GAT-heads 12 12 12 --GAT-hidden-sizes 768 768 --dropout-prob 0.3 --skip-connections linear
gat_subparser = subparsers.add_parser('GAT')
gat_subparser.set_defaults(GNN_model='GATConv')
gat_subparser.add_argument('--GNN-hidden-sizes', type=int, nargs='+', default=(1024, 1024), help='hidden layer sizes of the GNN model.')
gat_subparser.add_argument('--GAT-heads', type=int, nargs='+', default=(4, 4, 6), help='Number of multi-head-attentions in each layer of GAT. Ignored if using non-GAT GCN models.')
gat_subparser.add_argument('--GNN-activation', type=str, choices=activations, default='ELU', help='Activation module to use for the GNN layers.')
gat_subparser.add_argument('--skip-connections', type=str, default='linear', choices=('none', 'identity', 'linear'), help='type of skip connections. Identity: y = x + conv(x). Linear: y = lin(x) + conv(x)')
gat_subparser.add_argument('--norm-act-drop-before-conv', action='store_true', help='Perform normalization, activation and dropout before graph convolution.')

gin_subparser = subparsers.add_parser('GIN')
gin_subparser.set_defaults(GNN_model='GINConv')
gin_subparser.add_argument('--GNN-hidden-sizes', type=int, nargs='+', default=(1024, 1024), help='hidden layer sizes of the GNN model.')
gin_subparser.add_argument('--GIN-hidden-sizes', type=int, nargs='+', default=(1024, 512), help='hidden layer sizes of the MLP in the GIN model.')
gin_subparser.add_argument('--GIN-activation', type=str, choices=activations, default='ReLU', help='Activation module to use for the MLP in the GIN model.')
gin_subparser.add_argument('--GNN-activation', type=str, choices=activations, default='ReLU', help='Activation module to use for the GNN layers.')
gin_subparser.add_argument('--skip-connections', type=str, default='linear', choices=('none', 'identity', 'linear'), help='type of skip connections. Identity: y = x + conv(x). Linear: y = lin(x) + conv(x)')

gcn_subparser = subparsers.add_parser('GCN')
gcn_subparser.set_defaults(GNN_model='GCNConv')
gcn_subparser.add_argument('--GNN-hidden-sizes', type=int, nargs='+', default=(16,), help='hidden layer sizes of the GNN model.')
gcn_subparser.add_argument('--GCN-improved', action='store_true', help='If set to True, the layer computes A_hat as A+2I.')
gcn_subparser.add_argument('--no-GNN-normalize', action='store_false', dest='GNN_normalize', help='If set to True, will add self-loops and apply symmetric normalization. (default: True)')
gcn_subparser.add_argument('--GNN-activation', type=str, choices=activations, default='ReLU', help='Activation module to use for the GNN layers.')
gcn_subparser.add_argument('--skip-connections', type=str, default='none', choices=('none', 'identity', 'linear'), help='type of skip connections. Identity: y = x + conv(x). Linear: y = lin(x) + conv(x)')

sage_subparser = subparsers.add_parser('SAGE')
sage_subparser.set_defaults(GNN_model='SAGEConv')
sage_subparser.add_argument('--GNN-hidden-sizes', type=int, nargs='+', default=(64,), help='hidden layer sizes of the GNN model.')
sage_subparser.add_argument('--GNN-normalize', action='store_true', help='If set to True, output features will be ℓ2-normalized.')
sage_subparser.add_argument('--GNN-activation', type=str, choices=activations, default='ReLU', help='Activation module to use for the GNN layers.')
sage_subparser.add_argument('--skip-connections', type=str, default='none', choices=('none', 'identity', 'linear'), help='type of skip connections. Identity: y = x + conv(x). Linear: y = lin(x) + conv(x)')

gcn2_subparser = subparsers.add_parser('GCN2')
gcn2_subparser.set_defaults(GNN_model='GCN2Conv', GNN_norm_module='layer')
gcn2_subparser.add_argument('--GNN-hidden-sizes', type=int, nargs='+', default=(256, 256, 256, 256, 256, 256, 256, 256, 256), help='hidden layer sizes of the GNN model.')
gcn2_subparser.add_argument('--GCN2-alpha', type=float, default=0.5, help='The strength of the initial residual connection α.')
gcn2_subparser.add_argument('--GCN2-theta', type=float, default=1., help='The hyperparameter θ to compute the strength of the identity mapping β = log(θ / l + 1).')
gcn2_subparser.add_argument('--GCN2-shared-weights', action='store_true', help='If set to False, will use different weight matrices for the smoothed representation PX and the initial residual X_0 (“GCNII*”).')
gcn2_subparser.add_argument('--no-GNN-normalize', action='store_false', dest='GNN_normalize', help='If set to True, will add self-loops and apply symmetric normalization. (default: True)')
gcn2_subparser.add_argument('--GNN-activation', type=str, choices=activations, default='ReLU', help='Activation module to use for the GNN layers.')
gcn2_subparser.add_argument('--skip-connections', type=str, default='identity', choices=('none', 'identity', 'linear'), help='type of skip connections. Identity: y = x + conv(x). Linear: y = lin(x) + conv(x)')

gen_subparser = subparsers.add_parser('DeeperGCN')
gen_subparser.set_defaults(GNN_model='GENConv', MLP_norm_module='layer', GNN_norm_module='layer', dropout_prob=0.3)
gen_subparser.add_argument('--GNN-hidden-sizes', type=int, nargs='+', default=(256, 256, 256, 256, 256, 256, 256, 256, 256), help='hidden layer sizes of the GNN model.')
gen_subparser.add_argument('--skip-connections', type=str, default='identity', choices=('none', 'identity', 'linear'), help='type of skip connections. Identity: y = x + conv(x). Linear: y = lin(x) + conv(x)')
gen_subparser.add_argument('--GNN-activation', type=str, choices=activations, default='ReLU', help='Activation module to use for the GNN layers.')
gen_subparser.add_argument('--no-GEN-learn-temp', action='store_false', dest='GEN_learn_temp', help='If true, learn the softmax temparature in the aggregation function of GEN.')
gen_subparser.add_argument('--norm-act-drop-after-conv', action='store_false', dest='norm_act_drop_before_conv', help='Perform normalization, activation and dropout before graph convolution.')

unet_subparser = subparsers.add_parser('GraphUNet')
unet_subparser.set_defaults(GNN_model='GraphUNet')
unet_subparser.add_argument('--GNN-hidden-sizes', type=int, nargs='+', default=(64, 64, 64), help='hidden layer sizes of the GNN model.')
unet_subparser.add_argument('--skip-connections', type=str, default='identity', choices=('none', 'identity', 'linear'), help='type of skip connections. Identity: y = x + conv(x). Linear: y = lin(x) + conv(x)')

crf_subparser = subparsers.add_parser('CRF')
crf_subparser.set_defaults(GNN_model='CRF', no_proxy=True, no_log_softmax=True)

# belief propagation params
parser.add_argument('--max-prod-bp-steps', type=int, default=5, help='max-product belief propagation iteration steps')
parser.add_argument('--sum-prod-bp-steps', type=int, default=2, help='sum-product belief propagation iteration steps')
parser.add_argument('--edge-pred-softmax-temp', type=float, default=10, help='Softmax temperature of pred_edge logit: turn down this to get smoother prediction distribution')
parser.add_argument('--edge-pred-softmax-temp-candidates', type=float, nargs='+', default=(100, 10, 2, 1, 0.5, 0.2, 0.1), help='Candidate softmax temperatures of pred_edge logit: turn down this to get smoother prediction distribution')
parser.add_argument('--edge-marginal-norm-coef', type=float, default=0.1, help='Weight of node marginals to normalize edge marginal into edge potential.')
parser.add_argument('--marginal-softmax-temp', type=float, default=1, help='Softmax temperature of p_s and p_st logit: turn down this to get smoother marginal distribution')
parser.add_argument('--edge-appearance-prob', type=float, default=0.5, help='Appearance probability of each edge used in the tree-reweighted BP.')
