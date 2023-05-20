"""Convert a PyG dataset to a format that is acceptable by cfgnn."""
import argparse
import pickle

from sklearn.model_selection import train_test_split
import torch_geometric as pyg

parser = argparse.ArgumentParser()
parser.add_argument(
    "--class", "-c", dest="class_", type=str,
    help="Name (case sensitive) of PyG class. For example, Planetoid for Cora."
)
parser.add_argument(
    "--dataset", "-d", dest="dataset", type=str,
    help="Name (case sensitive) of the pyg dataset. For example, Cora."
)
parser.add_argument(
    "--path", "-p", dest="path", type=str, default=None,
    help="Where to store the pyg dataset or where is it stored if already downloaded."
)
parser.add_argument(
    "--output", "-o", dest="out", type=str, default="./",
    help="Where to store the cfgnn dataset."
)
parser.add_argument(
    "--test_split", "-t", dest="test_split", type=float, default=0.1,
    help="Test split in case the dataset does not have a train and test mask."
)

args = parser.parse_args()
print(args)
if args.path is None:
    args.path = args.dataset
try:
    dataset = eval(f"pyg.datasets.{args.class_}(root='{args.path}', name='{args.dataset}')")
    data = dataset[0]
except:
    print("Invalid dataset name.")
    exit(1)


#* -----> Dataset stats.
print(f"Dataset: {dataset}:")
print("=" * 75)
print(f"Number of graphs: {len(dataset)}")
print(f"Number of features: {dataset.num_features}")
print(f"Number of classes: {dataset.num_classes}")

print()
print(data)
print("=" * 75)

# Gather some statistics about the graph.
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
print(f"Has isolated nodes: {data.has_isolated_nodes()}")
print(f"Has self-loops: {data.has_self_loops()}")
print(f"Is undirected: {data.is_undirected()}")
try:
    print(f"Number of training nodes: {data.train_mask.sum()}")
    print(f"Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}")
except:
    print("No training mask.")


#* -----> Convert the dataset to cfgnn format.
print("\nGenerating dataset...")
dense_adj = pyg.utils.to_dense_adj(data.edge_index).numpy()
features = data.x.unsqueeze(0).numpy()
labels = data.y.unsqueeze(0).numpy()
if "train_mask" in dir(data):
    train_mask = (data.train_mask == 1).nonzero().flatten().tolist()
    val_mask = (data.val_mask == 1).nonzero().flatten().tolist()
else:
    train_indices, test_indices, train_labels, test_labels = train_test_split(
        list(range(dense_adj.shape[-1])), labels[0], stratify=labels[0]
    )


#* -----> Save to disk.
cfgnn = dict()
cfgnn["adj"] = dense_adj
cfgnn["feat"] = features
cfgnn["labels"] = labels
cfgnn["train_idx"] = train_indices
cfgnn["test_idx"] = test_indices

print("Writing to disk...")
with open(f"{args.out}/{args.dataset}.pickle", "wb") as file:
    pickle.dump(cfgnn, file)
print("Done!")
