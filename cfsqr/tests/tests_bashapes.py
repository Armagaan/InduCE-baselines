"""Node classification baselines BAShapes"""

## * Imports
from collections import defaultdict
import pickle
import sys

import numpy as np
import torch

sys.path.append(".")
from models.gcn import GCNNodeBAShapes
from utils.preprocessing.ba_shapes_preprocessing import \
    ba_shapes_preprocessing

if len(sys.argv) != 2:
    print(f"Usage: python tests.py [FOLDER]")
    print("FOLDER: Specify the folder containing the outputs of get_outputs.sh.")
    exit(1)

# Specify the folder containing the outputs of get_outputs.sh.
FOLDER = sys.argv[1]

with open(f"{FOLDER}/exp_dict.pkl", "rb") as file:
    exp_dict = pickle.load(file) # format: node_id: explanation_mask over the adjacency_matrix

with open(f"{FOLDER}/log.txt", "r") as file:
    log = file.readlines()

with open(f"{FOLDER}/num_dict.pkl", "rb") as file:
    num_dict = pickle.load(file) # format: node_id: #counterfactuals_found

with open(f"{FOLDER}/pred_label_dict.pkl", "rb") as file:
    pred_label_dict = pickle.load(file) # format: node_id: initial_blackbox_prediction

with open(f"{FOLDER}/new_label_dict.pkl", "rb") as file:
    new_label_dict = pickle.load(file) # format: node_id: new_blackbox_prediction

with open(f"{FOLDER}/pred_proba.txt", "r") as file:
    pred_proba = file.readlines()

with open(f"{FOLDER}/t_gid.pkl", "rb") as file:
    t_gid = pickle.load(file) # format: subgraph_id (same as node_id)

## * Constants
NUMBER_OF_LABELS = len(str.strip(pred_proba[0]).split())

NODES_PER_LABEL = defaultdict(int)
for node_id, label in pred_label_dict.items():
    NODES_PER_LABEL[f"label-{int(label)}"] += 1

print(f"Nodes per label: {dict(NODES_PER_LABEL)}")

# predictions = defaultdict(int)
# for node_id, line in zip(t_gid, pred_proba):
#     line = line.strip().split()
#     line = [float(pred) for pred in line]
#     predictions[node_id] = line.index(max(line))

## * Per-label Explanation size
per_label_explanation_size = defaultdict(list)

# iterate over the nodes
for node_id, number_of_cfs in num_dict.items():
    # find out the initial label
    label = pred_label_dict[node_id]
    # update size of corresponding label
    if label != new_label_dict[node_id] and number_of_cfs != 0:
        per_label_explanation_size[f"label-{int(label)}"].append(int(number_of_cfs))

# find mean and std
numerator = list()
denominator = list()
for label in range(NUMBER_OF_LABELS):
    if len(per_label_explanation_size[f"label-{int(label)}"]) == 0:
        mean, std = None, None
    else:
        for i in per_label_explanation_size[f"label-{int(label)}"]:
            numerator.append(i)
        denominator.append(len(per_label_explanation_size[f"label-{int(label)}"]))
        mean = np.mean(per_label_explanation_size[f"label-{int(label)}"])
        std = np.std(per_label_explanation_size[f"label-{int(label)}"])
    per_label_explanation_size[f"label-{int(label)}"] = [mean, std]

print("\nPer-label explanation size:")
for key, value in per_label_explanation_size.items(): # format: label: (mean, std)
    print(f"{key}: {value[0]} +- {value[1]}")


## * Explanation size
# mean = np.array(list(num_dict.values())).mean()
# std = np.array(list(num_dict.values())).std()
# explanation_size = [mean, std]
# print(f"\nExplanation size:\n{explanation_size[0]} +- {explanation_size[1]}")
if len(denominator) != 0:
    mean = sum(numerator) / sum(denominator)
    std = np.std(np.array(numerator))
else:
    mean, std = None, None
explanation_size = [mean, std]
print(f"\nExplanation size:\n{explanation_size[0]} +- {explanation_size[1]}")

## * Per-node fidelity
labels_and_preds = defaultdict(tuple)
for node_id in t_gid:
    labels_and_preds[node_id] = (int(pred_label_dict[node_id]), new_label_dict[node_id])
per_label_cf_found = defaultdict(int)

for node_id, (label, prediction) in labels_and_preds.items():
    if label != prediction:
        per_label_cf_found[f"label-{label}"] += 1

per_label_fidelity = dict()
for key, value in per_label_cf_found.items():
    per_label_fidelity[key] = 1 - per_label_cf_found[key]/NODES_PER_LABEL[key]

print(f"\nPer-label fidelity:")
for key, value in per_label_fidelity.items():
    print(f"{key}: {value}")


## * Fidelity
cf_found = 0
for node_id, (label, prediction) in labels_and_preds.items():
    if label != prediction:
        cf_found += 1

fidelity = 1 - cf_found/sum(list(NODES_PER_LABEL.values()))
print(f"\nFidelity:\n{fidelity}")

## * Per label sparsity
print(f"\nPer-label sparsity:")
for label in range(NUMBER_OF_LABELS):
    sparcity_list = list()
    for node_id, number_of_cfs in num_dict.items():
        if pred_label_dict[node_id] != label or number_of_cfs == 0:
            continue
        adj = exp_dict[node_id]
        adj = 1 * (adj != 0) # Threshold defined by cfsqr's authors.
        num_edges = adj.sum().item() / 2
        sparcity_list.append(1 - (number_of_cfs / num_edges))
    if len(sparcity_list) != 0:
        sparcity = np.mean(sparcity_list)
    else:
        sparcity = None
    print(f"label {label}: {sparcity}")

## * Sparsity
sparcity_list = list()
for node_id, number_of_cfs in num_dict.items():
    adj = exp_dict[node_id]
    adj = 1 * (adj != 0) # Threshold defined by cfsqr's authors.
    num_edges = adj.sum().item() / 2
    sparcity_list.append(1 - (number_of_cfs / num_edges))
sparcity = np.mean(sparcity_list)
print(f"\nSparcity: {sparcity}")


## * Accurcy
G_dataset = ba_shapes_preprocessing(
    dataset_dir="datasets/BA_Shapes",
    hop_num=4
)

base_model = GCNNodeBAShapes(
        in_feats=G_dataset.feat_dim,
        h_feats=20,
        out_feats=20,
        num_classes=4,
        device='cpu',
        if_exp=True
    )
base_model.load_state_dict(
    torch.load("cfgnn_model_weights/gcn_3layer_syn1.pt")
)
base_model.eval()

cf_acc_list = list()
for gid, cf_adj in exp_dict.items():
    label = labels_and_preds[gid][0]
    pred = labels_and_preds[gid][1]
    if label == pred:
        continue
    adj_size = int(np.sqrt(cf_adj.size()))
    adj = cf_adj.reshape(adj_size, adj_size)
    adj = adj * (adj > 0.5) # Threshold defined by cfsqr's authors.
    cfs = adj.nonzero()
    cf_acc = 0
    # if no cf was found
    if cfs.size(0) == 0:
        continue
    for cf in cfs:
        src, dest = cf
        predictions = base_model(
                G_dataset.graphs[gid],
                G_dataset.graphs[gid].ndata['feat'].float(),
                G_dataset.graphs[gid].edata['weight'],
                G_dataset.targets[gid],
                testing=True
            ).argmax(dim=-1)
        src_pred, dest_pred = predictions[src], predictions[dest]
        if src_pred != 0 and dest_pred != 0:
            cf_acc += 1
    cf_acc = cf_acc / len(cfs)
    cf_acc_list.append(100 * cf_acc)
if len(cf_acc_list) != 0:
    cf_acc_mean = np.mean(cf_acc_list)
else:
    cf_acc_mean = None
print(f"CF accuracy: {cf_acc_mean} %")
