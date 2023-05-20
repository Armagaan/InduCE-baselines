#!/bin/bash

# * Run this script prior to running the notebooks.
# * The output of this script is needed in those notebooks.

# ! Run this script from the root directory of the repo.

usage() {
    # echo correct usage to cosole
    echo
    echo "USAGE: bash ${0} [-d DATASET] [-e EVALMDOE]" >&2
    echo
    echo "DATASET:  Dataset name. One of [Mutagenicity, NCI1, IsCyclic]"
    echo "EVALMODE: Test on seen or unseen graphs. One of [eval, train]"
    exit 1
}

# Check whether the required number of arguments are supplied.
if [ ${#} != 4 ]; then
    usage
fi

# Parse command line arguments
while getopts "d:e:" OPTION; do
    case ${OPTION} in
        d) DATASET=${OPTARG} ;;
        e) EVALMODE=${OPTARG} ;;
        ?) usage ;;
    esac
done
shift "$(( OPTIND - 1 ))"

DATASETS="Mutagenicity NCI1 IsCyclic"
EVALMODES="eval train"
MIN=0.0
MAX=1.0
if [[ ! " ${DATASETS[*]} " =~ " ${DATASET} " ]]; then
    echo "Invalid dataset"
    usage
elif [[ ! " ${EVALMODES[*]} " =~ " ${EVALMODE} " ]]; then
    echo "Invalid evalmode"
    usage
fi

# Activate the conda environment.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gem

# Choose a script based on the supplied dataset.
echo
case ${DATASET} in
    Mutagenicity)
        echo -e "\n******* DISTILLATION *******\n"
        python generate_ground_truth_graph_classification.py --dataset=Mutagenicity --output=mutag --graph-mode --top_k=20 --evalmode="$EVALMODE"
        echo -e "\n******* TRAINING *******\n"
        python explainer_gae_graph.py --distillation=mutag_top20 --output=mutag_top20 --dataset=Mutagenicity --gpu -b 128 --weighted --gae3 --loss=mse --early_stop --graph_labeling --train_on_positive_label --epochs=300 --lr=0.01 --evalmode="$EVALMODE" >/dev/null
        ;;
    NCI1)
        echo -e "\n******* DISTILLATION *******\n"
        python generate_ground_truth_graph_classification.py --dataset=NCI1 --output=nci1_dc --graph-mode --top_k=20 --disconnected --evalmode="$EVALMODE"
        echo -e "\n******* TRAINING *******\n"
        python explainer_gae_graph.py --distillation=nci1_dc_top20 --output=nci1_dc_top20 --dataset=NCI1 --gpu -b 128 --weighted --gae3 --loss=mse --early_stop --graph_labeling --train_on_positive_label --epochs=300 --lr=0.01 --evalmode="$EVALMODE" >/dev/null
        ;;
    IsCyclic)
        echo -e "\n******* DISTILLATION *******\n"
        python generate_ground_truth_graph_classification.py --dataset=IsCyclic --output=iscyclic --graph-mode --top_k=20 --evalmode="$EVALMODE"
        echo -e "\n******* TRAINING *******\n"
        python explainer_gae_graph.py --distillation=iscyclic_top20 --output=iscyclic_top20 --dataset=IsCyclic --gpu -b 128 --weighted --gae3 --loss=mse --early_stop --graph_labeling --train_on_positive_label --epochs=300 --lr=0.01 --evalmode="$EVALMODE" >/dev/null
        ;;
    *)
        echo "Something's wrong" >&2
        exit 1
        ;;
esac

echo -e "\n******* BASELINES *******\n"
python tests/baselines_graph.py "$DATASET" "$EVALMODE"

exit 0
