# OGBG Arxiv
import pickle
from math import ceil
from time import perf_counter

import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch.nn.functional import dropout, relu
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import PGExplainer
from torch_geometric.explain.config import ModelConfig
from torch_geometric.explain.metric import fidelity
from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm

np.random.seed(42)
torch.manual_seed(42)

# Explainer parameters.
EPOCHS = 20
LR = 0.003
TOP_K = 4
print(f"Epochs {EPOCHS}, Top_K {TOP_K}")

# * Data
dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="../data/")
graph = dataset[0] # pyg graph object
features = graph.x
edge_index = graph.edge_index
labels = graph.y.flatten()
print(graph)
print(f'Average node degree: {graph.num_edges / graph.num_nodes:.2f}')
print(f'Has isolated nodes: {graph.has_isolated_nodes()}')

split_idx = dataset.get_idx_split()
idx_train_sampled = list()
for i in range(40):
    idx = (graph.y[split_idx['train']] == i).nonzero(as_tuple=False)
    idx_sub = np.random.choice(idx[:,0].numpy(), size=10, replace=False)
    idx_train_sampled.extend(idx_sub)

idx_train_full = split_idx['train']
idx_train = torch.tensor(idx_train_sampled, dtype=torch.long)
idx_val = split_idx['valid']
idx_test_full = split_idx['test']

with open("../data/eval-sets/ogbg-arxiv.pickle", "rb") as file:
    eval_indices = pickle.load(file)
eval_indices = sorted([i.item() for i in eval_indices])
idx_test = split_idx['test'][eval_indices]

# * Model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for __ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        return

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = relu(x)
            x = dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

HIDDEN = 256
LAYERS = 3
DROPOUT = 0.5
LAYERS = 3

model = GCN(
    in_channels=128,
    hidden_channels=HIDDEN,
    out_channels=40,
    num_layers=LAYERS,
    dropout=DROPOUT
)
model.load_state_dict(torch.load(f"../models/gcn_3layer_ogbg-arxiv.pt"))

def test(indices):
    model.eval()
    with torch.no_grad():
        output = model(graph.x, graph.edge_index)
        pred = torch.argmax(output, dim=1)
    return (pred[indices] == graph.y.flatten()[indices]).sum().item() / len(pred[indices])

print(f"Acc train (full): {test(idx_train_full):.4f}")
print(f"Acc train: {test(idx_train):.4f}")
print(f"Acc val: {test(idx_val):.4f}")
print(f"Acc test (full): {test(idx_test_full):.4f}")
print(f"Acc test (eval): {test(idx_test):.4f}")

# * Explain
explainer = Explainer(
    model=model,
    algorithm=PGExplainer(epochs=EPOCHS, lr=LR),
    explanation_type="phenomenon",
    edge_mask_type="object",
    model_config=ModelConfig(
        mode="multiclass_classification",
        task_level="node",
        return_type="log_probs"
    ),
    threshold_config=dict(threshold_type='topk', value=TOP_K),
)
print()
print("Training...")
for epoch in range(EPOCHS):
    print("-" * 10)
    print(f"Epoch {epoch}")
    for index in tqdm(idx_train):
        loss = explainer.algorithm.train(
            epoch=epoch,
            model=model,
            x=graph.x,
            edge_index=graph.edge_index,
            target=graph.y.flatten(),
            index=index.item()
        )

# * Metrics
start = perf_counter()
for index in idx_test:
    explanation = explainer(graph.x, graph.edge_index, target=graph.y.flatten(), index=index.item())
end = perf_counter()
print("Time elapsed:", round(end - start, 2), "seconds")

## Fidelity
def cal_fidelity(indices):
    fidelities = list()
    for index in indices:
        explanation = explainer(features, edge_index, target=labels, index=index.item())
        fidelities.append(fidelity(explainer, explanation)[0])
    fidelities = torch.tensor(fidelities, dtype=float)
    return fidelities.mean(), fidelities.std()

fidelity_mean_train, fidelity_std_train = cal_fidelity(idx_train)
fidelity_mean_test, fidelity_std_test = cal_fidelity(idx_test)

print(f"Average training fidelity: {1 - fidelity_mean_train:.4f}, std={fidelity_std_train:.4f}")
print(f"Average test fidelity: {1 - fidelity_mean_test:.4f}, std={fidelity_std_test:.4f}")

## Size
print(f"Average explanaiton size: {TOP_K}")

## Sparsity
def cal_sparsity(indices, is_undirected:bool = True):
    sparsity_t = list()
    # extract the subgraph
    for node_index in indices:
        __, __, __, edge_mask = k_hop_subgraph(
            node_idx=node_index.item(),
            num_hops=PGExplainer._num_hops(model),
            edge_index=edge_index,
            num_nodes=features.size(0),
            flow=PGExplainer._flow(model),
            directed=False, # * change according to dataset.
        )
        # find the number of edges in the subgraph.
        num_edges = edge_mask.nonzero().size(0)
        # account for undirected edges.
        # if is_undirected:
        #     num_edges = ceil(num_edges / 2)
        if num_edges < TOP_K:
            continue
        else:
            sparsity = 1 - (TOP_K / num_edges)
            sparsity_t.append(sparsity)
    sparsity_t = torch.tensor(sparsity_t, dtype=float)
    return sparsity_t.mean(), sparsity_t.std()

sparsity_mean_train, sparsity_std_train = cal_sparsity(idx_train, True)
sparsity_mean_test, sparsity_std_test = cal_sparsity(idx_test, True)

print(f"Average train sparsity: {sparsity_mean_train:.4f}, std={sparsity_std_train:.4f}")
print(f"Average test sparsity: {sparsity_mean_test:.4f}, std={sparsity_std_test:.4f}")
