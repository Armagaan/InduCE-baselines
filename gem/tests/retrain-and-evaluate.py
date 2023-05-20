"""Compute node classification baselines.
- Run this script after activating "gem" conda environment.
- This script assumes that you have already computed the distillations.
- Do not run multiple scripts on the same dataset parallelly! They will overwrite each other's explanations.
"""

import os
import sys

if len(sys.argv) != 3:
    print("\nUSAGE: python tests/baseline_script.py [DATASET] [EVALSET]")
    print("[DATASET]: syn1, syn4, syn5")
    print("[EVALSET]: eval, train")
    exit(1)
DATASET = sys.argv[1]
EVALSET = sys.argv[2]

if DATASET not in ['syn1', 'syn4', 'syn5']:
    print("\nINVALID DATASET!")
    print("[DATASET]: syn1, syn4, syn5")
    exit(1)
if EVALSET not in ['eval', 'train']:
    print("\nINVALID EVALSET")
    print("[EVALSET]: eval, train")
    exit(1)

# Remove old explanations.
print(f"This will delete the following folder: explanation/{DATASET}_top6")
go = input("Do you want to proceed? [y or n] ")
assert go in ["y", "n"], "Invalid input."
if go == "n":
    exit(0)
os.system(f"rm -r explanation/{DATASET}_top6")

# Train the explainer.
print("\nTraining started!")
os.system(
    f"python explainer_gae.py "
    f"--dataset={DATASET} "
    f"--distillation={DATASET}_top6 "
    f"--output={DATASET}_top6 "
    f"--gpu "
    f"--evalset={EVALSET} "
    f"1> /dev/null"
)
print("Training complete!")
# Run this script to store the sub_adjacencies.
os.system(
    f"python test_explained_adj.py "
    f"--dataset={DATASET} "
    f"--distillation={DATASET}_top6 "
    f"--exp_out={DATASET}_top6 "
    f"--top_k=6 "
    f"--evalset={EVALSET} "
    # f"1> /dev/null"
)
print("\nSaved sub_adjacencies!")
# Compute the baselines.
OUTPUT = sorted(os.listdir(f"output/{DATASET}"))[-1]
print("\n-----> Baselines <-----")
os.system(f"python tests/baselines.py {DATASET} {EVALSET} output/{DATASET}/{OUTPUT}")
