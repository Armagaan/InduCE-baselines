# Baselines

## Imports
from collections import defaultdict
import pickle
import sys

import numpy as np
import torch
from torch_geometric.utils import \
    k_hop_subgraph,\
    dense_to_sparse,\
    to_dense_adj,\
    subgraph

sys.path.append("src/utils")
from utils import get_neighbourhood

if len(sys.argv) != 2:
    print("USAGE: python test.py [dataset]")
    print("dataset: one of [bashapes, treecycles, treegrids, small_amazon, cora, citeseer]")
    exit(1)

# specify the datset
DATASET = sys.argv[1]

if DATASET == "bashapes":
    path_cfs = "results/syn1/random/syn1_epochs500"
    path_predictions = "results/syn1/random/predictions.pkl"
    path_eval_set = "data/Eval-sets/eval-set-bashapes.pkl"
    path_data = "data/gnn_explainer/syn1.pickle"

elif DATASET == "treecycles":
    path_cfs = "results/syn4/random/syn4_epochs500"
    path_predictions = "results/syn4/random/predictions.pkl"
    path_eval_set = "data/Eval-sets/eval-set-treecycles.pkl"
    path_data = "data/gnn_explainer/syn4.pickle"

elif DATASET == "treegrids":
    path_cfs = "results/syn5/random/syn5_epochs500"
    path_predictions = "results/syn5/random/predictions.pkl"
    path_eval_set = "data/Eval-sets/eval-set-treegrids.pkl"
    path_data = "data/gnn_explainer/syn5.pickle"

elif DATASET == "small_amazon":
    path_cfs = "results/small_amazon/random/small_amazon_epochs500"
    path_predictions = "results/small_amazon/random/predictions.pkl"
    path_eval_set = "data/Eval-sets/eval-set-small_amazon.pkl"
    path_data = "data/gnn_explainer/small_amazon.pickle"

else:
    print("Invalid dataset!")
    exit(1)

with open(path_data, "rb") as file:
	data = pickle.load(file)

adj = torch.Tensor(data["adj"]).squeeze() # Does not include self loops
features = torch.Tensor(data["feat"]).squeeze()
labels = torch.tensor(data["labels"]).squeeze()
idx_train = torch.tensor(data["train_idx"])
edge_index = dense_to_sparse(adj)

with open(path_cfs, "rb") as file:
    cfs = pickle.load(file)

with open(path_predictions, "rb") as file:
    predictions = pickle.load(file)

with open(path_eval_set, "rb") as file:
    eval_set = pickle.load(file)


## Constants
if DATASET == "bashapes":
    NUMBER_OF_LABELS = 4
elif DATASET == "treecycles" or DATASET == "treegrids":
    NUMBER_OF_LABELS = 2
else:
    NUMBER_OF_LABELS = len(torch.unique(labels))
PREDICTIONS = {node:int(prediction) for node, prediction in enumerate(predictions)}
NODES_PER_PREDICTED_LABEL = defaultdict(int)
for node in PREDICTIONS:
    label = PREDICTIONS[node]
    NODES_PER_PREDICTED_LABEL[f"label-{label}"] += 1
PREDICTIONS_EVAL_SET = {node:label for node, label in PREDICTIONS.items() if node in eval_set}

NODES_PER_PREDICTED_LABEL_IN_EVAL_SET = defaultdict(int)
for node in PREDICTIONS_EVAL_SET:
    label = PREDICTIONS_EVAL_SET[node]
    NODES_PER_PREDICTED_LABEL_IN_EVAL_SET[f"label-{label}"] += 1

CAP = 1e10
# if DATASET == "small_amazon":
#     CAP = 12

#* Per-label explanation size
# have a dictionary for each label
per_label_explanation_size = defaultdict(list)
nodes_per_prediction = defaultdict(int)

# iterate over the cfs
for cf in cfs:
    # if cf wasn't found, skip to next iteration
    if len(cf) == 0 or cf[-1] == 0 or cf[-1] > CAP or cf[4] == cf[5]:
        continue
    original_prediction = cf[5]
    # just get cfs[-1][11] (which is the #edge-deletions)
    perturbations = cf[9]
    # store this against the corresponding label in the dictionry
    per_label_explanation_size[f"label-{int(original_prediction)}"].append(int(perturbations))

for label in per_label_explanation_size:
    nodes_per_prediction[label] = len(per_label_explanation_size[label])

for label in range(NUMBER_OF_LABELS):
    # if there was no node in the eval-set with that label
    if len(per_label_explanation_size[f"label-{int(label)}"]) == 0:
        mean, std = None, None
    else:
        mean = np.mean(per_label_explanation_size[f"label-{int(label)}"])
        std = np.std(per_label_explanation_size[f"label-{int(label)}"])
    per_label_explanation_size[f"label-{int(label)}"] = [mean, std]
print("\n==========")
print("Per-label Explanation size:")
for key, value in per_label_explanation_size.items(): # format: label: (mean, std)
    print(f"{key}: {value[0]} +- {value[1]}")
print()
print(f"Nodes per predicted label in the eval-set:\n{NODES_PER_PREDICTED_LABEL_IN_EVAL_SET}")
print(f"Nodes per post-perturbation-prediction in the eval-set:\n{nodes_per_prediction}")


## Explanation size
explanation_size = list()
missed = 0
# iterate over the cfs
for cf in cfs:
    # if cf wasn't found, hence skip
    if len(cf) == 0 or cf[-1] == 0 or cf[-1] > CAP or cf[4] == cf[5]:
        missed += 1
        continue
    explanation_size.append(int(cf[9]))
# take mean and std
explanation_size = [np.mean(explanation_size), np.std(explanation_size)]
print("\n====================")
print("Explanation_size:")
print(f"{explanation_size[0]:.2f} +- {explanation_size[1]:.2f}")
print()
print(f"#Nodes in the eval set: {len(eval_set)}")
print(f"#Nodes for which cf wasn't found: {missed}")
print(f"Hence, #nodes over which size was calculated: {len(eval_set) - missed}")


#* Per-label Fidelity
nodes_for_which_cf_was_found = [cf[0] for cf in cfs if cf[4] != cf[5] and cf[-1] != 0 and cf[-1] <= CAP]
per_label_misses = defaultdict(int)

# iterate over cfs
for node in eval_set:
    # get prediction
    label = PREDICTIONS[node]
    # check if cf was found
    if node not in nodes_for_which_cf_was_found:
        per_label_misses[f"label-{label}"] += 1

per_label_fidelity = defaultdict(int)
for label in per_label_misses:    
    per_label_fidelity[label] = per_label_misses[label]/NODES_PER_PREDICTED_LABEL_IN_EVAL_SET[label]
print("\n====================")
print("Per label fidelity:")
print(per_label_fidelity)


## Fidelity
print("\n====================")
print("Fidelity:")
# fidelity = 1 - len(nodes_for_which_cf_was_found)/len(eval_set)
fidelity = sum(list(per_label_misses.values())) / len(eval_set)
print(f"{fidelity:.2f}")


## Accuracy
cf_acc_list = list()
# Iterate over the nodes in the eval set
for cf in cfs:
    if len(cf) == 0 or cf[-1] == 0 or cf[-1] > CAP:
        continue
    target_node = cf[0]
    cf_adj = cf[2]
    sub_adj = cf[3]
    # Extract the neighborhood.
    data_sub_graph = get_neighbourhood(
        node_idx=target_node,
        edge_index=edge_index,
        n_hops=4,
        features=features,
        labels=labels,
    )
    # Get the node mapping from sub_graph to full_graph.
    __, __, __, node_dict = data_sub_graph
    # Only possible because the values of the node_dict are all unique.
    reverse_node_dict = {val:key for key, val in node_dict.items()}
    # ! CFs are double counted here: (s,d), (d,s)
    cf_sources, cf_destinations = (sub_adj - cf_adj).nonzero()
    # Iterate over CFs
    cf_acc = 0
    for i in range(len(cf_sources)):
        # get original indices of the mapped src and dest nodes.
        src = reverse_node_dict[cf_sources[i]]
        dest = reverse_node_dict[cf_destinations[i]]
        # compute cf accuracy
        if labels[src] != 0 and labels[dest] != 0:
            cf_acc += 1
    cf_acc = cf_acc / (len(cf_sources))
    cf_acc_list.append(100 * cf_acc)
cf_acc_mean = np.mean(cf_acc_list)

print("\n====================")
print(f"CF accuracy: {cf_acc_mean:.2f} %")
