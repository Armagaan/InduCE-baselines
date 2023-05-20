from __future__ import division
from __future__ import print_function
import argparse
import pickle
import time
import sys
sys.path.append('..')

import numpy as np
import torch
import torch.nn.functional as F

from train import GCNSynthetic
from utils.utils import \
	normalize_adj,\
	get_neighbourhood,\
	safe_open,\
	get_degree_matrix,\
	create_symm_matrix_from_vec,\
	create_vec_from_symm_matrix
from torch_geometric.utils import dense_to_sparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='syn1')

# Based on original GCN models -- do not change
parser.add_argument('--hidden', type=int, default=20, help='Number of hidden units.')
parser.add_argument('--n_layers', type=int, default=3, help='Number of convolutional layers.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (between 0 and 1)')

# For explainer
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--num_epochs', type=int, default=500, help='Num epochs for explainer')
parser.add_argument('--device', default='cpu', help='CPU or GPU.')
args = parser.parse_args()

print(args)

args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.autograd.set_detect_anomaly(True)

# Import dataset from GNN explainer paper.
with open(f"data/gnn_explainer/{args.dataset}.pickle", "rb") as file:
	data = pickle.load(file)

adj = torch.Tensor(data["adj"]).squeeze() # Does not include self loops.
features = torch.Tensor(data["feat"]).squeeze()
labels = torch.tensor(data["labels"]).squeeze()
idx_train = torch.tensor(data["train_idx"])

if args.dataset == "syn1":
	filename_ = "bashapes"
elif args.dataset == "syn4":
	filename_ = "treecycles"
elif args.dataset == "syn5":
	filename_ = "treegrids"
elif args.dataset is not None:
	filename_ = args.dataset

with open(f"data/Eval-sets/eval-set-{filename_}.pkl", "rb") as file:
	idx_test = torch.tensor(pickle.load(file))

edge_index = dense_to_sparse(adj) # Needed for pytorch-geo functions.

# Change to binary task: 0 if not in house, 1 if in house.
if args.dataset == "syn1_binary":
	labels[labels==2] = 1
	labels[labels==3] = 1

# According to reparam trick from GCN paper.
norm_adj = normalize_adj(adj)

# Set up original model, get predictions
model = GCNSynthetic(
	nfeat = features.shape[1],
	nhid = args.hidden,
	nout = args.hidden,
	nclass = len(labels.unique()),
	dropout = args.dropout
)

model.load_state_dict(torch.load(f"models/gcn_3layer_{args.dataset}.pt"))
model.eval()
output = model(features, norm_adj)
y_pred_orig = torch.argmax(output, dim=1)
# Confirm model is actually doing something
print(f"y_true counts: {np.unique(labels.numpy(), return_counts=True)}")
print(f"y_pred_orig counts: {np.unique(y_pred_orig.numpy(), return_counts=True)}")

# Get CF examples in test set
test_cf_examples = []
start = time.time()
for i in idx_test[:]:
	best_loss = np.inf

	for n in range(args.num_epochs):
		sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(
			node_idx = int(i),
			edge_index = edge_index,
			n_hops = args.n_layers + 1,
			features = features,
			labels = labels,
		)
		new_idx = node_dict[int(i)]

		# Get CF adj, new prediction
		num_nodes = sub_adj.shape[0]

		# P_hat needs to be symmetric 
		# ==> learn vector representing entries in upper/lower triangular matrix
		# and use to populate P_hat later
		P_vec_size = int((num_nodes * num_nodes - num_nodes) / 2) + num_nodes

		# Randomly initialize P_vec in [-1, 1]
		r1 = -1
		r2 = 1
		P_vec = torch.FloatTensor((r1 - r2) * torch.rand(P_vec_size) + r2)
		P_hat_symm = create_symm_matrix_from_vec(P_vec, num_nodes) # Ensure symmetry
		P = (F.sigmoid(P_hat_symm) >= 0.5).float() # threshold P_hat

		# Get cf_adj, compute prediction for cf_adj
		cf_adj = P * sub_adj
		A_tilde = cf_adj + torch.eye(num_nodes)

		D_tilde = get_degree_matrix(A_tilde)
		# Raise to power -1/2, set all infs to 0s
		D_tilde_exp = D_tilde ** (-1 / 2)
		D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

		# Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
		cf_norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

		pred_cf = torch.argmax(model(sub_feat, cf_norm_adj), dim=1)[new_idx]
		pred_orig = torch.argmax(model(sub_feat, normalize_adj(sub_adj)), dim=1)[new_idx]
		# Number of edges changed (symmetrical)
		loss_graph_dist = sum(sum(abs(cf_adj - sub_adj))) / 2
		print(
			f"Node idx: {i}, "
			f"original pred: {pred_orig}, "
			f"cf pred: {pred_cf}, "
			f"graph loss: {loss_graph_dist}"
		)

		if (pred_cf != pred_orig) & (loss_graph_dist < best_loss):
			best_loss = loss_graph_dist
			print("best loss: {}".format(best_loss))
			best_cf_example = [
				i.item(),
				new_idx.item(),
				cf_adj.detach().numpy(),
				sub_adj.detach().numpy(),
				pred_cf.item(),
				pred_orig.item(),
				sub_labels[new_idx].numpy(),
				sub_adj.shape[0],
				node_dict,
				loss_graph_dist.item()
			]
	test_cf_examples.append(best_cf_example)
	print(
		f"Time for {args.num_epochs} epochs of one example: "
		f"{(time.time() - start)/60:.4f}min"
	)

print(f"Total time elapsed: {(time.time() - start)/60:.4f}min")

# Save CF examples in test set
with safe_open(
	f"results"
	f"/{args.dataset}"
	f"_baseline_cf_examples_epochs{args.num_epochs}", "wb"
) as file:
	pickle.dump(test_cf_examples, file)

with safe_open(
	f"results"
	f"/{args.dataset}"
	f"/random"
	f"/{args.dataset}"
	f"_epochs{args.num_epochs}", "wb"
) as f:
	pickle.dump(test_cf_examples, f)

with safe_open(
	f"results"
	f"/{args.dataset}"
	f"/random"
	f"/predictions.pkl", "wb"
) as file:
	pickle.dump(y_pred_orig, file)
