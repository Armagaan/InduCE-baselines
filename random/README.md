# Random deletions

# Environment
```sh
conda env create -f env.yml
conda activate cfgnn
```

# Run
```sh
bash src/get_outputs.sh -d [dataset]

python tests/tests.py [dataset]
```
Datasets: bashapes, treecycles, treegrids, small_amazon.

# New dataset
Say, you want to compute the baselines on a dataset named Cora.
- Create `data/gnn_explainer/cora.pickle`.
- Create `data/Eval-sets/eval-set-cora.pkl`
- Create `models/gcn_3layer_cora.pt`.
- Update `tests/get_outputs.sh`.
- Update `tests/tests.py`.
