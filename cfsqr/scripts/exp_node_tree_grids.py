import os
import sys
from time import time

import numpy as np
import torch

from utils.argument import arg_parse_exp_node_tree_grids
from models.explainer_models import NodeExplainerEdgeMulti
from models.gcn import GCNNodeTreeGrids
from utils.preprocessing.tree_grids_preprocessing import \
    TreeGridsDataset, tree_grids_preprocessing

if __name__ == "__main__":
    torch.manual_seed(1000)
    np.random.seed(0)
    np.set_printoptions(threshold=sys.maxsize)
    exp_args = arg_parse_exp_node_tree_grids()
    if exp_args.edge_additions == "True":
        exp_args.edge_additions = True
    else:
        exp_args.edge_additions = False
    print("argument:\n", exp_args)
    model_path = exp_args.model_path
    # train_indices = np.load(os.path.join(model_path, 'train_indices.pickle'), allow_pickle=True)

    # * START: Updated by Burouj: 17 May, 22
    # Author's eval set
    # test_indices = np.load(os.path.join(model_path, 'test_indices.pickle'), allow_pickle=True)
    
    # Our eval set
    test_indices = np.load('datasets/Eval-sets/eval-set-treegrids.pkl', allow_pickle=True)
    # * END

    G_dataset = tree_grids_preprocessing(dataset_dir="datasets/Tree_Grids", hop_num=4)
    # G_dataset = TreeCyclesDataset(load_path=os.path.join(model_path))
    # targets = np.load(os.path.join(model_path, 'targets.pickle'), allow_pickle=True)  # the target node to explain
    graphs = G_dataset.graphs
    labels = G_dataset.labels
    targets = G_dataset.targets
    if exp_args.gpu:
        device = torch.device('cuda:%s' % exp_args.cuda)
    else:
        device = 'cpu'
    base_model = GCNNodeTreeGrids(
        in_feats=G_dataset.feat_dim,
        h_feats=20,
        out_feats=20,
        num_classes=2,
        device=device,
        if_exp=True
    ).to(device)
    base_model.load_state_dict(torch.load("cfgnn_model_weights/gcn_3layer_syn5.pt"))
    
    #  fix the base model
    # * Very important: If not done the blackbox weights start altering during explanation.
    for param in base_model.parameters():
        param.requires_grad = False
    # * Fixes the dropout layer
    base_model.eval()

    predictions = list()
    for gid in test_indices:
        probability = base_model(
            g=graphs[gid],
            in_feat=graphs[gid].ndata["feat"].float(),
            # e_weight=edge_weights,
            e_weight=graphs[gid].edata["weight"],
            target_node=targets[gid]
        )[0]
        predictions.append(torch.argmax(probability))
    predictions = torch.Tensor(predictions)
    print(predictions.unique(return_counts=True))
    # exit(0)

    # Create explainer
    explainer = NodeExplainerEdgeMulti(
        base_model=base_model,
        G_dataset=G_dataset,
        args=exp_args,
        test_indices=test_indices,
        # fix_exp=6
    )
    FOLDER_PATH = exp_args.output
    start_time = time()
    explainer.explain_nodes_gnn_stats(FOLDER_PATH)
    end_time = time()
    print(f"Time elapsed: {end_time - start_time:2f} seconds")
