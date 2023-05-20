"""Node Classification Baselines"""

# -----> Imports <-----
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append("gnnexp")
from models import GCNSynthetic

if len(sys.argv) != 7:
    print("\nUSAGE: python tests/baselines.py [DATASET] [EVALMODE] [OUTPUTS-FOLDER] [TOP_K] [HIDDEN_DIM] [OUT_DIM]")
    print("[DATASET]: syn1, syn4, syn5")
    print("[EVALMODE]: train, eval\n")
    print("[OUTPUTS-FOLDER]: path to folder containing the output of evaluate_adj.py")
    print("[TOP_K]: top_k used in distiallation and training.")
    print("[HIDDEN_DIM]: hidden dimension of the blackbox")
    print("[OUT_DIM]: Out dimension of the blackbox.")
    print("Hint: output/syn1/12345, output/syn4/12345, output/syn5/12345")
    exit(1)


# -----> Constants <-----
DATASET = sys.argv[1]
EVAL = sys.argv[2]
OUTPUTS = sys.argv[3]
TOP_K = int(sys.argv[4])
HIDDEN_DIM = int(sys.argv[5])
OUT_DIM = int(sys.argv[6])

if  DATASET not in ['syn1', 'syn4', 'syn5', 'small_amazon']:
    print("INVALID DATASET!")
    exit(1)
elif EVAL not in ['eval', 'train']:
    print("INVALID EVALMODE!")
    exit(1)


# -----> Data <-----
# The extracted subadjacency matrices.
with open(f"{OUTPUTS}/original_sub_data.pkl", "rb") as file:
    sub_data = pickle.load(file)
sub_labels = dict()
for node in sub_data:
    new_idx = sub_data[node]['node_idx_new']
    sub_labels[node] = int(sub_data[node]['sub_label'][new_idx])

explanations = dict()
PATH = f"explanation/{DATASET}_top{TOP_K}"
for filename in os.listdir(PATH):
    if 'pred' not in filename:
        continue
    node_idx = ''.join(filter(lambda i: i.isdigit(), filename))
    explanations[int(node_idx)] = pd.read_csv(
        f"{PATH}/{filename}", header=None).to_numpy()


# -----> Model <-----
ckpt = torch.load(f"data/{DATASET}/eval_as_{EVAL}.pt")
cg_dict = ckpt["cg"]
input_dim = cg_dict["feat"].shape[2]
num_classes = cg_dict["pred"].shape[2]
feat = torch.from_numpy(cg_dict["feat"]).float()
adj = torch.from_numpy(cg_dict["adj"]).float()
label = torch.from_numpy(cg_dict["label"]).long()
# with open(f"tests/prog_args_{DATASET}.pkl", "rb") as file:
#     prog_args = pickle.load(file)

model = GCNSynthetic(
    nfeat=input_dim,
    nhid= HIDDEN_DIM, #prog_args.hidden_dim,
    nout= OUT_DIM, #prog_args.output_dim,
    nclass=num_classes,
    dropout=0.0,
)
model.load_state_dict(ckpt["model_state"])
model.eval()


# -----> Accuracy <-----
feat_cfgnn = torch.from_numpy(ckpt['cg']["feat"]).float()
adj_cfgnn = torch.from_numpy(ckpt['cg']["adj"]).float()
label_cfgnn = torch.from_numpy(ckpt['cg']["label"]).long()
preds_cfgnn = model(feat_cfgnn, adj_cfgnn)
predicted_labels_cfgnn = torch.argmax(preds_cfgnn, dim=-1)
acc = 100 * torch.sum(predicted_labels_cfgnn == label_cfgnn).item() / label_cfgnn.size(1)
print(f"Accuracy on full graph: {acc:.2f} %")


# -----> Top k edges of the explanation <-----
top_indices = dict() # They are from the Upper Triangular part only.
top_k = TOP_K
for graph_id, graph in explanations.items():
    # print(graph_id, graph.shape)
    # triu: Upper Triangular
    # abs: These are indices of the flattended version, not of the 2D version.
    # * Following what gem's authors have done:
    # * explanation = adjecency * explanation_mask
    explanation = sub_data[graph_id]['org_adj'].squeeze(0) * graph
    triu_abs_top_indices = (-np.triu(explanation).flatten()).argsort()[:top_k]
    index_rows = triu_abs_top_indices // explanation.shape[0]
    index_cols = triu_abs_top_indices % explanation.shape[0]
    triu_top_k_indices = [(r,c) for r,c in zip(index_rows, index_cols)]
    top_indices[graph_id] = triu_top_k_indices


# -----> Fidelity, Explanation size, and CF Accuracy <-----
explanation_size_list = list()
cf_found_count = 0
cf_acc_list = list()
sparsity_list = list()
acc = 0
correctly_classified_nodes = 0
for graph_id, graph in explanations.items():
    feat = sub_data[graph_id]["sub_feat"].float()
    adj = sub_data[graph_id]["org_adj"].float()
    label = sub_labels[graph_id]
    new_node_id = sub_data[graph_id]['node_idx_new']
    # get original prediction
    original_prediction = int(model(feat, adj).squeeze(0).argmax(dim=-1)[new_node_id])
    if original_prediction == label:
        acc += 1
    # Set size_count, cf_acc to 0.
    # Make a copy of the original adjacency.
    size_count = 0
    cf_acc = 0
    new_adj = adj.clone()
    # Work on correctly predicted nodes only.
    if label == 0:
        continue
    go = False
    if label == original_prediction:
        go = True
    if not go:
        continue
    correctly_classified_nodes += 1
    for i, index in enumerate(top_indices[graph_id]):
        r1, c1 = index
        r2, c2 = c1, r1 # for the lower triangular part.
        # Remove the edges.
        new_adj[0][r1, c1] = 0.0
        new_adj[0][r2, c2] = 0.0
        # Make the prediction.
        new_prediction = int(
            model(feat, new_adj).squeeze(0).argmax(dim=-1)[new_node_id]
        )
        # print(label, new_prediction)
        # Increase size_count by 1.
        size_count += 1
        # Compute cf accuracy
        src, dest = r1, c1
        src_label = sub_data[graph_id]['sub_label'][src]
        dest_label = sub_data[graph_id]['sub_label'][dest]
        if src_label != 0 and dest_label != 0:
            cf_acc += 1
        # If the label flipped: stop
        # if original_prediction != new_prediction:
        #     cf_found_count += 1
        #     explanation_size_list.append(size_count)
        #     cf_acc_list.append(100 * cf_acc / (i + 1))
        #     break
    # print()
    explanation_size_list.append(size_count)
    sparsity_list.append(1 - (size_count / (adj.sum() / 2)))
    cf_acc_list.append(100 * cf_acc / (i + 1))
    if original_prediction != new_prediction:
        cf_found_count += 1

acc = 100 * acc/len(explanations)
print(f"Accuracy on subgraphs: {acc:.2f} %")
try:
    fidelity = 1 - cf_found_count/correctly_classified_nodes
except ZeroDivisionError:
    fidelity = None
    print("No correctly classified nodes.", end="")
if len(explanation_size_list) == 0:
    exp_size_mean = None
    exp_size_std = None
    cf_acc_mean = None
else:
    exp_size_mean = np.mean(explanation_size_list)
    exp_size_std = np.std(explanation_size_list)
    cf_acc_mean = np.mean(cf_acc_list)
sparsity = np.mean(sparsity_list)

print(f"\nFidelity: {fidelity}")
print(f"Explanation size: {exp_size_mean}, std={exp_size_std}")
print(f"CF accuracy: {cf_acc_mean:.2f} %")
print(f"Sparsity: {sparsity:.4f}")
print()

# -----> Per-label Fidelity, Explanation size, and CF Accuracy <-----
if DATASET == 'syn1':
    labels = [1,2,3]
elif DATASET == "small_amazon":
    labels = list(range(9))
else:
    labels = [1]
for desired_label in labels:
    explanation_size_list = list()
    cf_found_count = 0
    cf_acc_list = list()
    correctly_classified_nodes = 0
    sparsity_list = list()
    for graph_id, graph in explanations.items():
        feat = sub_data[graph_id]["sub_feat"].float()
        adj = sub_data[graph_id]["org_adj"].float()
        label = sub_labels[graph_id]
        new_node_id = sub_data[graph_id]['node_idx_new']
        # Get original prediction.
        original_prediction = int(model(feat, adj).squeeze(0).argmax(dim=-1)[new_node_id])
        # Set size_count, cf_acc to 0.
        # Make a copy of the original adjacency.
        size_count = 0
        cf_acc = 0
        new_adj = adj.clone()
        # Work on correctly predicted nodes only.
        if label != desired_label:
            continue
        go = False
        # print(f"Correct? : label: {label} | pred: {original_prediction}")
        if label == original_prediction:
            go = True
        if not go:
            continue
        correctly_classified_nodes += 1
        for i, index in enumerate(top_indices[graph_id]):
            r1, c1 = index
            r2, c2 = c1, r1 # for the lower triangular part.
            # Remove the edges.
            new_adj[0][r1, c1] = 0.0
            new_adj[0][r2, c2] = 0.0
            # make the prediction
            new_prediction = int(
                model(feat, new_adj).squeeze(0).argmax(dim=-1)[new_node_id]
            )
            # Increase the size_count by 1.
            size_count += 1
            # Compute the cf accuracy.
            src, dest = r1, c1
            src_label = sub_data[graph_id]['sub_label'][src]
            dest_label = sub_data[graph_id]['sub_label'][dest]
            if src_label != 0 and dest_label != 0:
                cf_acc += 1
            # If the label flipped: stop
            # if original_prediction != new_prediction:
            #     cf_found_count += 1
            #     explanation_size_list.append(size_count)
            #     cf_acc_list.append(100 * cf_acc / (i + 1))
            #     break
        explanation_size_list.append(size_count)
        sparsity_list.append(1 - (size_count / (adj.sum() / 2)))
        cf_acc_list.append(100 * cf_acc / (i + 1))
        # print(f"CF? : orig: {original_prediction} | new: {new_prediction}")
        if original_prediction != new_prediction:
            cf_found_count += 1

    print(f"\n-----> Label-{desired_label}")
    try:
        fidelity = 1 - cf_found_count/correctly_classified_nodes
    except ZeroDivisionError:
        fidelity = None
        print("No correctly classified nodes.")
    if len(explanation_size_list) == 0:
        exp_size_mean = None
        exp_size_std = None
        cf_acc_mean = None
        print("No cf found!")
    else:
        exp_size_mean = np.mean(explanation_size_list)
        exp_size_std = np.std(explanation_size_list)
        cf_acc_mean = np.mean(cf_acc_list)
    sparsity = np.mean(sparsity_list)
    print(f"Number of nodes: {len(explanation_size_list)}")
    print(f"Fidelity: {fidelity}")
    print(f"Explanation size: {exp_size_mean}, std={exp_size_std}")
    print(f"CF accuracy: {cf_acc_mean} %")
    print(f"Sparsity: {sparsity}")
    print()
