#!/bin/bash

# Activate habitat conda environment.
. activate habitat

# Train habitat baselines example agent on example data.
cd /habitat-lab
python -u habitat_baselines/run.py --exp-config $1 --run-type train $2

