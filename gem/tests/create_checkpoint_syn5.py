"""Create a checkpoint for TreeGrids dataset AKA. syn5.
USAGE: python tests/create_checkpoint_syn5.py
"""

import os
import pickle
import sys

import torch

sys.path.append("gnnexp")
from models import GCNSynthetic


# ===== DATA =====
filename = "../cfgnnexplainer/data/gnn_explainer/syn5.pickle"
with open(filename, "rb") as file:
    treegrids_cfgnn = pickle.load(file)


# ===== MODEL =====
model = GCNSynthetic(
    nfeat=10,
    nhid=20,
    nout=20,
    nclass=2,
    dropout=0.0
)
state_dict = torch.load("cfgnn_model_weights/gcn_3layer_syn5.pt")
model.load_state_dict(state_dict)
model.eval()


# ===== PREDICTIONS =====
feat = torch.from_numpy(treegrids_cfgnn['feat']).float()
adj = torch.from_numpy(treegrids_cfgnn['adj']).float()
preds = model(feat, adj).detach().numpy()


#  ===== CHECKPOINT =====
ckpt_bashapes = torch.load("data/syn1/eval_as_eval.pt")
ckpt_treegrids = ckpt_bashapes.copy()
ckpt_treegrids['model_state'] = state_dict
ckpt_treegrids['cg'] = treegrids_cfgnn.copy()
ckpt_treegrids['cg']['pred'] = preds # This key is missing in treegrids_cfgnn.
ckpt_treegrids['cg']['label'] = ckpt_treegrids['cg'].pop('labels') # Key mismatch.


# ===== EVAL SET =====
with open("../eval_set.pkl", "rb") as file:
    eval_set = pickle.load(file)
KEY = "syn5/tree-grid"

# Our eval set as part of the training set
train_set_indices = [i for i in range(treegrids_cfgnn['labels'].shape[1])]
test_set_indices = list(set(eval_set[KEY]))
ckpt_treegrids["cg"]["train_idx"] = train_set_indices
ckpt_treegrids["cg"]["test_idx"] = test_set_indices
os.makedirs(f"data/syn5", exist_ok=True)
torch.save(ckpt_treegrids, f"data/syn5/eval_as_train.pt")

# Our eval set as the validation set
train_set_indices = [
    i for i in range(treegrids_cfgnn['labels'].shape[1])
    if i not in eval_set[KEY]
]
test_set_indices = list(set(eval_set[KEY]))
ckpt_treegrids["cg"]["train_idx"] = train_set_indices
ckpt_treegrids["cg"]["test_idx"] = test_set_indices
os.makedirs(f"data/syn5", exist_ok=True)
torch.save(ckpt_treegrids, f"data/syn5/eval_as_eval.pt")

print("Checkpoint saved in the folder: data/syn5")
