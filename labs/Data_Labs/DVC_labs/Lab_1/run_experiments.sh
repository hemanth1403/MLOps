#!/bin/bash
# ============================================
# run_experiments.sh - Run multiple DVC experiments
#
# Demonstrates how DVC tracks different model
# configurations and lets you compare results.
# ============================================

set -e

echo "=========================================="
echo "  Running DVC Experiments"
echo "=========================================="

# --- Experiment 1: Random Forest (default) ---
echo ""
echo ">>> Experiment 1: Random Forest (default params)"
dvc repro
git add .
git commit -m "Experiment 1: Random Forest baseline"
git tag -a exp-rf-baseline -m "Random Forest baseline experiment"
dvc push

echo ""
echo "Results for Experiment 1:"
dvc metrics show

# --- Experiment 2: Gradient Boosting ---
echo ""
echo ">>> Experiment 2: Gradient Boosting"

# Swap model type in params.yaml using sed
sed -i 's/model_type: "random_forest"/model_type: "gradient_boosting"/' params.yaml

dvc repro
git add .
git commit -m "Experiment 2: Gradient Boosting"
git tag -a exp-gb -m "Gradient Boosting experiment"
dvc push

echo ""
echo "Results for Experiment 2:"
dvc metrics show

# --- Experiment 3: Logistic Regression ---
echo ""
echo ">>> Experiment 3: Logistic Regression"

sed -i 's/model_type: "gradient_boosting"/model_type: "logistic_regression"/' params.yaml

dvc repro
git add .
git commit -m "Experiment 3: Logistic Regression"
git tag -a exp-lr -m "Logistic Regression experiment"
dvc push

echo ""
echo "Results for Experiment 3:"
dvc metrics show

# --- Compare all experiments ---
echo ""
echo "=========================================="
echo "  Comparing All Experiments"
echo "=========================================="
dvc metrics diff exp-rf-baseline exp-lr

echo ""
echo "To see full diff between any two experiments:"
echo "  dvc metrics diff <tag1> <tag2>"
echo ""
echo "To revert to a previous experiment:"
echo "  git checkout <tag> && dvc checkout"
