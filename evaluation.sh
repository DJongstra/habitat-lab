#!/bin/bash

########################################################################################################################
# Python (conda) environment.
########################################################################################################################

# Activate habitat conda environment.
echo "Activating habitat environment."
. activate habitat

########################################################################################################################
# Setup wandb.
########################################################################################################################

# Wandb
echo "Setting up wandb."

# Update locale, otherwise the Click package used by wandb complains (python 3.6).
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Create /root/.netrc file for wandb (otherwise wandb pull will not work)
echo "machine api.wandb.ai" >> /root/.netrc
echo "  login user" >> /root/.netrc
echo "  password $WANDB_API_KEY" >> /root/.netrc

########################################################################################################################
# Download run from wandb.
########################################################################################################################

# Download experiment data from wandb.
mkdir -p /tmp/wandb
cd /tmp/wandb

echo "Downloading experiment data from wandb."
wandb pull --entity $WANDB_ENTITY --project $WANDB_PROJECT_EVAL $WANDB_RUN_ID_EVAL

########################################################################################################################
# Run evaluation.
########################################################################################################################

# Train habitat baselines example agent on example data.
cd /habitat-lab
python -u habitat_baselines/run.py --exp-config $1 --run-type eval $2 EVAL_CKPT_PATH_DIR /tmp/wandb/checkpoints

