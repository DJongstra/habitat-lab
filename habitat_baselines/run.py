#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import random

import numpy as np
import torch

from habitat import make_dataset
from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config

import wandb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def execute_exp(config: Config, run_type: str) -> None:
    r"""This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    runtype: str {train or eval}
    """
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    if config.FORCE_TORCH_SINGLE_THREADED and torch.cuda.is_available():
        torch.set_num_threads(1)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config, runtype=run_type)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()

# PATHS_TO_JUNK = {
#     'LOG_FILE': '/private/home/naokiyokoyama/junk/train.log',
#     'CHECKPOINT_FOLDER': '/private/home/naokiyokoyama/junk',
#     'TENSORBOARD_DIR': '/private/home/naokiyokoyama/junk',
#     'VIDEO_DIR': '/private/home/naokiyokoyama/junk',
# }

def run_exp(exp_config: str, run_type: str, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    # if 'JUNK' in opts and opts[opts.index('JUNK')+1] == 'True':
    #     for k,v in PATHS_TO_JUNK.items():
    #         if k in opts:
    #             opts[opts.index(k)+1] = v
    #         else:
    #             opts.extend([k, v])

    config = get_config(exp_config, opts)

    # Replicate environments to support multiple PPO workers.
    config = get_replica_config(config)

    # Setup weights-and-biases monitoring.
    print("Setting up wandb monitoring")
    config.defrost()
    config.LOG_FILE = os.path.join(wandb.run.dir, f'{run_type}.log')
    config.CHECKPOINT_FOLDER = os.path.join(wandb.run.dir, 'checkpoints')
    config.TENSORBOARD_DIR = os.path.join(wandb.run.dir, 'tb')
    config.VIDEO_DIR = os.path.join(wandb.run.dir, 'videos')

    config.freeze()

    # Save config to wandb.
    save_config(exp_config, config)

    # Execute experiment.
    execute_exp(config, run_type)


def save_config(exp_config, config):
    print("Saving config using wandb.")
    # Save the original configs.
    wandb.save(exp_config)
    wandb.save(config.BASE_TASK_CONFIG_PATH)

    # Log hyper-parameters (find a better way of doing this).
    wandb.config.update({
        "num_processes": config.NUM_PROCESSES,
        "num_updates": config.NUM_UPDATES,
        "rl": {
            "ddppo": {
                "backbone": config.RL.DDPPO.backbone,
                "num_recurrent_layers": config.RL.DDPPO.num_recurrent_layers,
                "pretrained": config.RL.DDPPO.pretrained,
                "pretrained_encoder": config.RL.DDPPO.pretrained_encoder,
                "pretrained_weights": config.RL.DDPPO.pretrained_weights,
                "reset_critic": config.RL.DDPPO.reset_critic,
                "rnn_type": config.RL.DDPPO.rnn_type,
                "sync_frac": config.RL.DDPPO.sync_frac,
                "train_encoder": config.RL.DDPPO.train_encoder
            },
            "ppo": {
                "clip_param": config.RL.PPO.clip_param,
                "entropy_coef": config.RL.PPO.entropy_coef,
                "eps": config.RL.PPO.eps,
                "gamma": config.RL.PPO.gamma,
                "hidden_size": config.RL.PPO.hidden_size,
                "lr": config.RL.PPO.lr,
                "max_grad_norm": config.RL.PPO.max_grad_norm,
                "num_mini_batch": config.RL.PPO.num_mini_batch,
                "num_steps": config.RL.PPO.num_steps,
                "ppo_epoch": config.RL.PPO.ppo_epoch,
                "tau": config.RL.PPO.tau,
                "use_gae": config.RL.PPO.use_gae,
                "use_linear_clip_decay": config.RL.PPO.use_linear_clip_decay,
                "use_linear_lr_decay": config.RL.PPO.use_linear_lr_decay,
                "use_normalized_advantage": config.RL.PPO.use_normalized_advantage,
                "value_loss_coef": config.RL.PPO.value_loss_coef
            },
            "reward_measure": config.RL.REWARD_MEASURE,
            "slack_reward": config.RL.SLACK_REWARD,
            "success_measure": config.RL.SUCCESS_MEASURE,
            "success_reward": config.RL.SUCCESS_REWARD,
        },
        "gpulab": {
            "GPULAB_CLUSTER_ID": os.environ.get("GPULAB_CLUSTER_ID", ""),
            "GPULAB_CPUS_RESERVED": os.environ.get("GPULAB_CPUS_RESERVED", ""),
            "GPULAB_GPUS_RESERVED": os.environ.get("GPULAB_GPUS_RESERVED", ""),
            "GPULAB_JOB_ID": os.environ.get("GPULAB_JOB_ID", ""),
            "GPULAB_MEM_RESERVED": os.environ.get("GPULAB_MEM_RESERVED", ""),
            "GPULAB_PROJECT": os.environ.get("GPULAB_PROJECT", ""),
            "GPULAB_SLAVE_DNSNAME": os.environ.get("GPULAB_SLAVE_DNSNAME", ""),
            "GPULAB_SLAVE_HOSTNAME": os.environ.get("GPULAB_SLAVE_HOSTNAME",""),
            "GPULAB_SLAVE_INSTANCE_ID": os.environ.get("GPULAB_SLAVE_INSTANCE_ID", ""),
            "GPULAB_SLAVE_PID": os.environ.get("GPULAB_SLAVE_PID", ""),
            "GPULAB_USERNAME": os.environ.get("GPULAB_USERNAME", ""),
            "GPULAB_VERSION": os.environ.get("GPULAB_VERSION", ""),
            "GPULAB_WORKER_ID": os.environ.get("GPULAB_WORKER_ID", "")
        }
    })


def get_replica_config(config):
    r""" Updates the config to replicate scenes.
    Updates the config to include replicated scenes to support multiple workers
    when there are insufficient workers.
    :param config: Habitat.config
    :return: Habitat.config
    """

    print("Replica config")
    # Determine the number of scenes in this dataset.
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    if len(scenes) >= config.NUM_PROCESSES:
        # Fallback to default handling in habitat.
        return config

    assert config.NUM_PROCESSES % len(scenes) == 0, "Number of processes must be divisible by number of scenes."

    # Determine number of replicas.
    num_replicas = config.NUM_PROCESSES // len(scenes)

    # Update config.
    config.defrost()
    config.TASK_CONFIG.DATASET.CONTENT_SCENES = scenes * num_replicas
    config.freeze()

    return config


if __name__ == "__main__":
    # Setup weights-and-biases monitoring.
    wandb.init()

    main()
