# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from experiment_utils import cfg_dict_from_yaml
from rbc.cuda_manager import ConsumerFirmRunManagerBatchParallel


def check_if_ep_str_policy_exists(rollout_path, ep_str):
    return (
        rollout_path / Path("saved_models") / Path(f"consumer_policy_{ep_str}.pt")
    ).is_file()


def run_rollout(rollout_path, arguments):
    """
    # take in rollout directory
    # load latest policies and the action functions and the hparams dict
    # make a cudamanager obj and run the job
    # this will require initializing everything as before, resetting,
    # and running naive policy gradient training at some fixed learning rate
    """
    with open(rollout_path / Path("action_arrays.pickle"), "rb") as f:
        action_arrays = pickle.load(f)

    consumption_choices, work_choices, price_and_wage, tax_choices = (
        action_arrays["consumption_choices"],
        action_arrays["work_choices"],
        action_arrays["price_and_wage"],
        action_arrays["tax_choices"],
    )

    cfg_dict = cfg_dict_from_yaml(
        rollout_path / Path("hparams.yaml"),
        consumption_choices,
        work_choices,
        price_and_wage,
        tax_choices,
    )

    print(cfg_dict)

    if arguments.agent_type == "all":
        agent_types = ["consumer", "firm", "government"]
    else:
        agent_types = [arguments.agent_type]

    for agent_type in agent_types:
        ep_rewards = defaultdict(list)
        for _ in range(arguments.repeat_runs):
            for ep_str in arguments.ep_strs:
                if not check_if_ep_str_policy_exists(rollout_path, ep_str):
                    print(f"warning: {rollout_path} {ep_str} policy not found")
                    ep_rewards[ep_str].append([0.0])
                    continue
                m = ConsumerFirmRunManagerBatchParallel(cfg_dict)
                rewards_start = m.bestresponse_train(
                    agent_type,
                    arguments.num_episodes,
                    rollout_path,
                    ep_str=ep_str,
                    checkpoint=arguments.checkpoint_model,
                )
                ep_rewards[ep_str].append(rewards_start)

        with open(rollout_path / Path(f"br_{agent_type}_output.txt"), "w") as f:
            for ep_str in arguments.ep_strs:
                reward_arr = np.array(ep_rewards[ep_str])
                print(
                    f"mean reward (std) on rollout {ep_str}: "
                    f"before BR training {reward_arr[:,0].mean()} "
                    f"({reward_arr[:,0].std()}), "
                    f"after BR training {reward_arr[:,-1].mean()} "
                    f"({reward_arr[:,-1].std()}), "
                    f"mean improvement {(reward_arr[:,-1]-reward_arr[:,0]).mean()} "
                    f"({(reward_arr[:,-1]-reward_arr[:,0]).std()}",
                    file=f,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rolloutdir", type=str)
    parser.add_argument("num_episodes", type=int)
    parser.add_argument("--experiment-dir", action="store_true")
    parser.add_argument("--ep-strs", nargs="+", default=["0", "latest"])
    parser.add_argument("--agent-type", type=str, default="all")
    parser.add_argument("--repeat-runs", type=int, default=1)
    parser.add_argument("--checkpoint-model", type=int, default=100)

    args = parser.parse_args()

    if args.experiment_dir:
        exp_dir = Path(args.rolloutdir)
        for rolloutpath in exp_dir.iterdir():
            if rolloutpath.is_dir():
                run_rollout(rolloutpath, args)
    else:
        rolloutpath = Path(args.rolloutdir)
        run_rollout(rolloutpath, args)
