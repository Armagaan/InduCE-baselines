from __future__ import division, print_function

import sys

sys.path.append('..')
import argparse
import pickle
import time

import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse

from cf_explanation.cf_explainer import CFExplainer
from gcn import GCN
from utils.utils import get_neighbourhood, normalize_adj, safe_open

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='syn1')

# Based on original GCN models -- do not change
parser.add_argument('--hidden', type=int, default=20, help='Number of hidden units.')
#! This is never used in the model. The #layers is fixed to 3 for the blackbox and the explainer.
parser.add_argument('--n_layers', type=int, default=3, help='Number of convolutional layers.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (between 0 and 1)')

# For explainer
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for explainer')
parser.add_argument('--optimizer', type=str, default="SGD", help='SGD or Adadelta')
parser.add_argument('--n_momentum', type=float, default=0.0, help='Nesterov momentum')
parser.add_argument('--beta', type=float, default=0.5, help='Tradeoff for dist loss')
parser.add_argument('--num_epochs', type=int, default=500, help='Num epochs for explainer')
parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
parser.add_argument('--edge_additions', type=bool, default=False)
args = parser.parse_args()

print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.autograd.set_detect_anomaly(True)

# Import dataset from GNN explainer paper
with open("../data/gnn_explainer/{}.pickle".format(args.dataset), "rb") as f:
	data = pickle.load(f)

edge_index = data["adj"]
features = torch.Tensor(data["feat"]).squeeze()
labels = torch.tensor(data["labels"]).squeeze()
idx_train = torch.tensor(data["train_idx"])
with open(f"../data/Eval-sets/{args.dataset}.pickle", "rb") as file:
	idx_test = torch.tensor(pickle.load(file))

# Set up original model, get predictions
# HIDDEN = 256
# LAYERS = 3
# DROPOUT = 0.5
# LAYERS = 3

model = GCN(
    in_channels=128,
    hidden_channels=args.hidden,
    out_channels=40,
    num_layers=args.n_layers,
    dropout=args.dropout
)

model.load_state_dict(torch.load("../models/gcn_3layer_{}.pt".format(args.dataset)))
model.eval()

output = model(features, edge_index[0])
y_pred_orig = torch.argmax(output, dim=1)
# Confirm model is actually doing something.
print("y_true counts: {}".format(torch.unique(labels, return_counts=True)))
print("y_pred_orig counts: {}".format(torch.unique(y_pred_orig, return_counts=True)))

# Get CF examples in test set
test_cf_examples = []
start = time.time()
for i in idx_test[:5]:
	sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(int(i), edge_index, args.n_layers + 1, features, labels)
	new_idx = node_dict[int(i)]

	# Check that original model gives same prediction on full graph and subgraph
	# with torch.no_grad():
	# 	print("Output original model, full adj: {}".format(output[i]))
	# 	print("Output original model, sub adj: {}".format(model(sub_feat, normalize_adj(sub_adj))[new_idx]))

	# Need to instantitate new cf model every time because size of P changes based on size of sub_adj
	explainer = CFExplainer(
		model=model,
		sub_adj=sub_adj,
		sub_feat=sub_feat,
		n_hid=args.hidden,
		dropout=args.dropout,
		sub_labels=sub_labels,
		y_pred_orig=y_pred_orig[i],
		num_classes=len(labels.unique()),
		beta=args.beta,
		device=args.device,
		edge_additions=args.edge_additions,
	)

	if args.device == 'cuda':
		model.cuda()
		explainer.cf_model.cuda()
		features = features.cuda()
		labels = labels.cuda()
		idx_train = idx_train.cuda()
		idx_test = idx_test.cuda()

	cf_example = explainer.explain(
		node_idx=i,
		cf_optimizer=args.optimizer,
		new_idx=new_idx,
		lr=args.lr,
		n_momentum=args.n_momentum,
		num_epochs=args.num_epochs
	)
	test_cf_examples.append(cf_example)
	print("Time for {} epochs of one example: {:.4f}min".format(args.num_epochs, (time.time() - start)/60))

print("Total time elapsed: {:.4f}s".format((time.time() - start)/60))
print("Number of CF examples found: {}/{}".format(len(test_cf_examples), len(idx_test)))

# Save CF examples in test set.
with safe_open("../results/{}/{}/{}_cf_examples_lr{}_beta{}_mom{}_epochs{}_seed{}_edge_additions{}".format(
	args.dataset,
	args.optimizer,
	args.dataset,
	args.lr,
	args.beta,
	args.n_momentum,
	args.num_epochs,
	args.seed,
	args.edge_additions
), "wb") as f:
	pickle.dump(test_cf_examples, f)
