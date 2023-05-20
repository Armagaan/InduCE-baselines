# Baselines for node classification
- If you wish, use any GPU of your choice instead of 2.
- For using the CPU, remove the `--gpu` flag and `CUDA_VISIBLE_DEVICES=2`
- The distillations take up a lot of space and computation. In case you wish to recompute them, refer to `README.md` and delete/move precomputed distillations.
- In case you calculate the baselines multiple times. Delete the `explanation/[DATASET]` folder everytime you do so. Also, change the output folder path accordingly in step 3.

## Note
Say, you need to compute baselines on a new dataset named `small_amazonn`.
- Distillation
    - Create `data/small_amazon/eval_as_eval.pt` and `data/small_amazon/eval_as_train.pt` .
    - Run `generate_ground_truth.py` with `--top_k=k` . Choose `k` according to need. We need to do it for `12` and `22` .
- Train
    - Supply `--hidden-dim` and `--output-dim` appropriately.
- Test
    - Supply `--hidden-dim` and `--output-dim` appropriately.

## BA-Shapes
1. Train the classifier: `python explainer_gae.py --dataset=syn1 --distillation=syn1_top6 --output=syn1_top6 --evalset=[EVALSET] --gpu`
2. Save subgraphs: `python test_explained_adj.py --dataset=syn1 --distillation=syn1_top6 --exp_out=syn1_top6 --top_k=6 --evalset=[EVALSET]`
3. Compute baselines: `python tests/baselines.py syn1_top6 [EVALSET] output/syn1/<> [TOP_K] [HIDDEN_DIM] [OUT_DIM]`
    - Replace <> with the output folder which was generated after you ran `test_explained.py`
    - It would be the latest folder in the `output/syn1` folder.

## Tree-cycles
1. Train the classifier: `python explainer_gae.py --dataset=syn4 --distillation=syn4_top6 --output=syn4_top6 --evalset=[EVALSET] --gpu`
2. Save subgraphs: `python test_explained_adj.py --dataset=syn4 --distillation=syn4_top6 --exp_out=syn4_top6 --top_k=6 --evalset=[EVALSET]`
3. Compute baselines: `python tests/baselines.py syn4_top6 [EVALSET] output/syn4/<> [TOP_K] [HIDDEN_DIM] [OUT_DIM]`
    - Replace <> with the output folder which was generated after you ran `test_explained.py`
    - It would be the latest folder in the `output/syn4` folder.

##  Tree-grids
1. Train the classifier: `python explainer_gae.py --dataset=syn5 --distillation=syn5_top6 --output=syn5_top6 --evalset=[EVALSET] --gpu`
2. Save subgraphs: `python test_explained_adj.py --dataset=syn5 --distillation=syn5_top6 --exp_out=syn5_top6 --top_k=6 --evalset=[EVALSET]`
3. Compute baselines: `python tests/baselines.py syn5 [EVALSET] output/syn5/<> [TOP_K] [HIDDEN_DIM] [OUT_DIM]`
    - Replace <> with the output folder which was generated after you ran `test_explained.py`
    - It would be the latest folder in the `output/syn5` folder.
