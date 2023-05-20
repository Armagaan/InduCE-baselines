import os
import sys
from time import time

import numpy as np
import torch

from utils.argument import arg_parse_exp_node_citeseer
from models.explainer_models import NodeExplainerEdgeMulti
from models.gcn import GCNNodeCiteseer
from utils.preprocessing.citeseer_preprocessing import \
    CiteseerDataset, citeseer_preprocessing

if __name__ == "__main__":
    torch.manual_seed(1000)
    np.random.seed(0)
    np.set_printoptions(threshold=sys.maxsize)
    exp_args = arg_parse_exp_node_citeseer()
    print("argument:\n", exp_args)
    model_path = exp_args.model_path
    
    # Our eval set
    test_indices = np.load('datasets/Eval-sets/eval-set-citeseer.pkl', allow_pickle=True)

    G_dataset = citeseer_preprocessing(dataset_dir="datasets/citeseer", hop_num=4)

    graphs = G_dataset.graphs
    labels = G_dataset.labels
    targets = G_dataset.targets
    if exp_args.gpu:
        device = torch.device('cuda:%s' % exp_args.cuda)
    else:
        device = 'cpu'
    #todo Crosscheck these.
    base_model = GCNNodeCiteseer(
        in_feats=G_dataset.feat_dim,
        h_feats=64,
        out_feats=64,
        num_classes=8,
        device=device,
        if_exp=True
    ).to(device)
    base_model.load_state_dict(torch.load("cfgnn_model_weights/gcn_3layer_citeseer.pt"))
    
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
