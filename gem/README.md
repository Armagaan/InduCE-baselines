# Gem
## This repo covers the implementation for our paper Gem.

Wanyu Lin, Hao Lan, and Baochun Li. "[Generative Causal Explanations for Graph Neural Networks](https://arxiv.org/pdf/2104.06643.pdf)," in the Proceedings of the 38th International Conference on Machine Learning (ICML 2021), Online, July 18-24, 2021.

## Download code
```sh
git clone https://github.com/wanyu-lin/ICML2021-Gem Gem
cd Gem
git submodule init
git submodule update
cd gnnexp
git apply --ignore-space-change --ignore-whitespace ../gnnexp.patch
```

## Setup environment
Create an environment with conda:
```sh
conda env create -f env.yml
conda activate gem
```
If you face any issues, proceed with the following:
```sh
conda create -n gem python=3.8.8
conda activate gem
```
Install PyTorch with CUDA 10.2:
```sh
conda install pytorch cudatoolkit=10.2 -c pytorch
```
Or install PyTorch without CUDA:
```sh
conda install pytorch cpuonly -c pytorch
```
Install other required packages:
```sh
conda install opencv scikit-learn networkx pandas matplotlib seaborn
pip install tensorboardx
```

## Data, Distillation, Explanations, Output
To download the required data, precomputed distillations, outputs, and the explanations run:
```sh
python setup.py
```
It will download the above into appropriate folders.

## Distillation
```sh
python generate_ground_truth.py --dataset=syn1 --top_k=6
python generate_ground_truth.py --dataset=syn4 --top_k=6
python generate_ground_truth_graph_classification.py --dataset=Mutagenicity --output=mutag --graph-mode --top_k=20
python generate_ground_truth_graph_classification.py --dataset=NCI1 --output=nci1_dc --graph-mode --top_k=20 --disconnected
```
or you can directly extract from zip file
```
unzip distillation.zip
```

## Train Gem
```sh
python explainer_gae.py --dataset=syn1 --distillation=syn1_top6 --output=syn1_top6
python explainer_gae.py --dataset=syn4 --distillation=syn4_top6 --output=syn4_top6
python explainer_gae_graph.py --distillation=mutag_top20 --output=mutag_top20 --dataset=Mutagenicity --gpu -b 128 --weighted --gae3 --loss=mse --early_stop --graph_labeling --train_on_positive_label --epochs=300 --lr=0.01
python explainer_gae_graph.py --distillation=nci1_dc_top20 --output=nci1_dc_top20 --dataset=NCI1 --gpu -b 128 --weighted --gae3 --loss=mse --early_stop --graph_labeling --train_on_positive_label --epochs=300 --lr=0.01
```

## Evaluate Gem
```sh
python test_explained_adj.py --dataset=[DATASET] --distillation=[DATASET]_top[TOP_K] --exp_out=[DATASET]_top[TOP_K] --top_k=[TOP_K] --evalset=[EVALSET]

python tests/baselines.py syn4_top6 [EVALSET] output/[DATASET]/<> [TOP_K] [HIDDEN_DIM] [OUT_DIM]
```

For `syn1` with the pretrained model, testing in eval mode, it would look something like this:
```sh
python test_explained_adj.py --dataset=syn1 --distillation=syn1_top6 --exp_out=syn1_top6 --top_k=6 --evalset=[EVALSET]

python tests/baselines.py syn1 eval output/syn1/1660600121 6 20 20
```

For `syn4` with the pretrained model, testing in eval mode, it would look something like this:
```sh
python test_explained_adj.py --dataset=syn4 --distillation=syn4_top6 --exp_out=syn4_top6 --top_k=6 --evalset=[EVALSET]

python tests/baselines.py syn4 eval output/syn4/1681915247 6 20 20
```

For `syn5` with the pretrained model, testing in eval mode, it would look something like this:
```sh
python test_explained_adj.py --dataset=syn5 --distillation=syn5_top6 --exp_out=syn5_top6 --top_k=6 --evalset=[EVALSET]

python tests/baselines.py syn5 eval output/syn5/1660600177 6 20 20
```

For `small_amazon` with the pretrained model, testing in eval mode, it would look something like this:
```sh
python test_explained_adj.py --dataset=small_amazon --distillation=small_amazon_top12 --exp_out=small_amzon_top12 --top_k=12 --evalset=[EVALSET]

python tests/baselines.py small_amazon eval output/small_amazon/1671774946 12 64 64
```

## Visualization
Run `*.ipynb` files in Jupyter Notebook or Jupyter Lab.


## Reference
If you make advantage of Gem in your research, please cite the following in your manuscript:

```
@inproceedings{
    wanyu-icml21,
    title="{Generative Causal Explanations for Graph Neural Networks}",
    author={Lin, Wanyu and Lan, Hao and Li, Baochun},
    booktitle={International Conference on Machine Learning},
    year={2021},
    url={https://arxiv.org/pdf/2104.06643.pdf},
}
```
