# Based on https://github.com/tkipf/pygcn/blob/master/pygcn/train.py

from __future__ import division, print_function

import sys

sys.path.append('..')
import argparse
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from gcn import GCNSynthetic
from utils.utils import normalize_adj

# Defaults based on GNN Explainer
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='syn1')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=20, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--clip', type=float, default=2.0, help='Gradient clip.')
parser.add_argument('--device', default='cpu', choices=["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"], help='CPU or GPU.')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Import dataset from GNN explainer paper
with open("../data/gnn_explainer/{}.pickle".format(args.dataset), "rb") as f:
	data = pickle.load(f)

# For models trained using our GCN_synethic from GNNExplainer,
# using hyperparams from GNN explainer tasks
adj = torch.Tensor(data["adj"]).squeeze()
features = torch.Tensor(data["feat"]).squeeze()
labels = torch.tensor(data["labels"]).squeeze()
idx_train = torch.tensor(data["train_idx"])
idx_test = torch.tensor(data["test_idx"])

# Change to binary task: 0 if not in house, 1 if in house
if args.dataset == "syn1_binary":
	labels[labels==2] = 1
	labels[labels==3] = 1

norm_adj = normalize_adj(adj)

model = GCNSynthetic(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
                     nclass=len(labels.unique()), dropout=args.dropout)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if 'cuda' in args.device:
	torch.cuda.set_device(args.device)
	model.cuda()
	features = features.cuda()
	norm_adj = norm_adj.cuda()
	labels = labels.cuda()
	idx_train = idx_train.cuda()
	idx_test = idx_test.cuda()

def train(epoch):
	t = time.time()
	model.train()
	optimizer.zero_grad()
	output = model(features, norm_adj)
	loss_train = model.loss(output[idx_train], labels[idx_train])
	y_pred = torch.argmax(output, dim=1)
	acc_train = (y_pred[idx_train] == labels[idx_train]).sum() / len(y_pred[idx_train])
	loss_train.backward()
	clip_grad_norm_(model.parameters(), args.clip)
	optimizer.step()

	print('Epoch: {:04d}'.format(epoch+1),
		  'loss_train: {:.4f}'.format(loss_train.item()),
		  'acc_train: {:.4f}'.format(acc_train),
		  'time: {:.4f}s'.format(time.time() - t))

def test():
	model.eval()
	output = model(features, norm_adj)
	loss_test = F.nll_loss(output[idx_test], labels[idx_test])
	y_pred = torch.argmax(output, dim=1)
	acc_test = (y_pred[idx_test] == labels[idx_test]).sum() / len(y_pred[idx_test])
	print("Test set results:",
		  "loss= {:.4f}".format(loss_test.item()),
		  "accuracy= {:.4f}".format(acc_test))
	return y_pred

# Train model
t_total = time.time()
for epoch in range(args.epochs):
	train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

torch.save(model.state_dict(), "../models/gcn_3layer_{}".format(args.dataset) + ".pt")

# Testing
y_pred = test()

print("y_true counts: {}".format(torch.unique(labels, return_counts=True)))
print("y_pred_orig counts: {}".format(torch.unique(y_pred, return_counts=True)))
print("Finished training!")
