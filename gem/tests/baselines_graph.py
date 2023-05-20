"""Graph Classification Baselines"""


# -----> Imports <-----
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append("./")
from gnnexp.models import GNN_Custom_Mutag, GNN_Custom_NCI1, GNN_Custom_IsCyclic

def usage():
    print("\nUSAGE: python tests/baselines_grpah.py [DATASET] [EVALMODE]")
    print("[DATASET]: Mutagenicity, NCI1, IsCyclic")
    print("[EVALMODE]: eval, train\n")
    exit(1)

if len(sys.argv) != 3:
    usage()

DATASET = sys.argv[1] # OPTIONS: Mutagenicity, NCI1, IsCyclic
EVAL = sys.argv[2] # OPTIONS: eval, train
if DATASET not in ["Mutagenicity", "NCI1", "IsCyclic"]:
    print("INVALID DATASET!")
    usage()
if EVAL not in ["eval", "train"]:
    print("INVALID EVALMODE")
    usage()
#todo: MUTAG dataset is different from other baselines.
if DATASET == 'Mutagenicity':
    EXPLANATION_FOLDER = "mutag_top20"
    DESIRED_LABEL = 0
elif DATASET == 'NCI1':
    EXPLANATION_FOLDER = "nci1_dc_top20"
    DESIRED_LABEL = 1
elif DATASET == 'IsCyclic':
    EXPLANATION_FOLDER = "iscyclic_top20"
    DESIRED_LABEL = 1


# -----> Data <-----
explanations = dict()
PATH = f"explanation/{EXPLANATION_FOLDER}"
for filename in os.listdir(PATH):
    if 'pred' not in filename:
        continue
    graph_idx = ''.join(filter(lambda i: i.isdigit(), filename))
    explanations[int(graph_idx)] = pd.read_csv(f"{PATH}/{filename}", header=None).to_numpy()

ckpt = torch.load(f"ckpt/{DATASET}_eval_as_{EVAL}.pt")
cg_dict = ckpt["cg"] # get computation graph
input_dim = cg_dict["feat"].shape[2]


# -----> Model <-----
if DATASET == 'Mutagenicity':
    model = GNN_Custom_Mutag(
        in_features=input_dim,
        h_features=64,
    )
elif DATASET == 'NCI1':
    model = GNN_Custom_NCI1(
        in_features=input_dim,
        h_features=256,
    )
elif DATASET == 'IsCyclic':
    model = GNN_Custom_IsCyclic(
        in_features=input_dim,
        h_features=64,
    )
model.load_state_dict(ckpt["model_state"])
model.eval()


# -----> Predictions <-----
# predictions = list()
# labels = list()
# for graph_idx in cg_dict['test_idx']:
#     feat = cg_dict["feat"][graph_idx, :].float().unsqueeze(0)
#     adj = cg_dict["adj"][graph_idx].float().unsqueeze(0) # - explanations[graph_idx]
#     label = cg_dict["label"][graph_idx].float().unsqueeze(0)
#     proba = model(feat, adj)
#     predictions.append(proba.argmax(dim=-1))
#     labels.append(label)
# predictions = torch.Tensor(predictions)
# labels = torch.Tensor(labels)


# -----> Top-k edges of the explanation <-----
top_indices = dict() # They are from the Upper Triangular part only.
top_k = 20
for graph_id, graph in explanations.items():
    # print(graph_id, graph.shape)
    # triu: Upper Triangular
    # abs: These are indices of the flattended version, not of the 2D version.
    explanation = cg_dict['adj'][graph_id] * graph # following what gem's author's have done.
    triu_abs_top_indices = (-np.triu(explanation).flatten()).argsort()[:top_k]
    index_rows = triu_abs_top_indices // explanation.shape[0]
    index_cols = triu_abs_top_indices % explanation.shape[0]
    triu_top_k_indices = [(r,c) for r,c in zip(index_rows, index_cols)]
    top_indices[graph_id] = triu_top_k_indices


# -----> Fidelity & Explanation size <-----
explanation_size_list = list()
cf_found_count = 0
for graph_id, graph in explanations.items():
    feat = cg_dict["feat"][graph_id, :].float().unsqueeze(0)
    adj = cg_dict["adj"][graph_id].float().unsqueeze(0)
    label = cg_dict["label"][graph_id].long().unsqueeze(0)
    # get original prediction
    original_prediction = model(feat, adj).argmax(dim=-1)
    # set size_count to 0
    # make a copy of the original adjacency.
    size_count = 0
    new_adj = adj.clone()
    # work on correctly predicted label 1 nodes only.
    go = False
    if label == original_prediction == DESIRED_LABEL:
        go = True
    if not go:
        continue
    for index in top_indices[graph_id]:
        r1, c1 = index
        r2, c2 = c1, r1 # for the lower triangular part.
        # remove the edges
        new_adj[0][r1, c1] = 0.0
        new_adj[0][r2, c2] = 0.0
        # make the prediction
        new_prediction = model(feat, new_adj).argmax(dim=-1)
        # increase size_count by 1.
        size_count += 1
        # if the label flipped: stop
        if original_prediction != new_prediction:
            cf_found_count += 1
            explanation_size_list.append(size_count)
            break

fidelity = 1 - cf_found_count/len(explanations)
exp_size_mean = np.mean(explanation_size_list)
exp_size_std = np.std(explanation_size_list)

print(f"\nFidelity: {fidelity:.2f}")
print(f"Explanation size: mean={exp_size_mean:.2f}, std={exp_size_std:.2f}\n")
