# Compute baselines on a new dataset
*Note:* Please run all the scripts from the root folder of this repo.

Say, you need to compute baselines on a new dataset named `small_amazon`. You need to do the following:
- Create `scripts/exp_node_small_amazon.py` .
- Create `cfgnn_model_weights/gcn_3layer_small_amazon.pt` .
- Add `arg_parse_exp_node_small_amazon`  to `utils/argument.py`.
- Add `GCNNodeSmallAmazon` to `models/gcn.py` .
- Use `tests/cfgnn-to-cfsqr.ipynb` for:
    - Create `datasets/small_amazon/syn_data.pkl` .
    - Create `dataset/Eval-sets/eval-set-small_amazon.pkl` .
- Create `utils/preprocessing/small_amazon_preprocessing.py` .
- Create `outputs/small_amazon/` directory.
- Create `tests/tests_small_amazon.py` .
- Update `tests/get_outputs.sh` .
- Update `tests/baselines.sh` .
- Run `tests/get_outputs.sh` .
- Run `tests/small_amazon.py outputs/small_amazon/<folder-created>` .
