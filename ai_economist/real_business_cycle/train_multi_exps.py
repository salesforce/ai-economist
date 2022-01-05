# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import argparse
import os

from experiment_utils import (
    create_job_dir,
    run_experiment_batch_parallel,
    sweep_cfg_generator,
)
from rbc.constants import all_agents_short_export_experiment_template

train_param_sweeps = {
    "lr": [0.001],
    "entropy": [0.5],
    "batch_size": [128],
    "clip_grad_norm": [2.0],
    "base_seed": [2345],
    "should_boost_firm_reward": [False],
    "use_ppo": [True],
    "ppo_num_updates": [2, 4],
    "ppo_clip_param": [0.1],
}


agent_param_sweeps = {
    "consumer_lr_multiple": [1.0],
    "consumer_reward_scale": [5.0],
    "government_reward_scale": [5.0 * 100.0 * 2.0],
    "firm_reward_scale": [30000],
    "government_counts_firm_reward": [1],
    "government_lr_multiple": [0.05],
}


world_param_sweeps = {
    "initial_wages": [0.0],
    "interest_rate": [0.0],
    "importer_price": [500.0],
    "importer_quantity": [100.0],
    "use_importer": [1],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--experiment-dir", type=str, default="experiment/experiment")
    parser.add_argument("--group-name", type=str, default="default_group")
    parser.add_argument("--job-name-base", type=str, default="rollout")
    parser.add_argument("--num-consumers", type=int, default=100)
    parser.add_argument("--num-firms", type=int, default=10)
    parser.add_argument("--num-governments", type=int, default=1)
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--seed-from-timestamp", action="store_true")

    args = parser.parse_args()

    (
        default_cfg_dict,
        consumption_choices,
        work_choices,
        price_and_wage,
        tax_choices,
        default_firm_action,
        default_government_action,
    ) = all_agents_short_export_experiment_template(
        args.num_firms, args.num_consumers, args.num_governments
    )

    if args.run_only:
        print("Not sweeping over hyperparameter combos...")
    else:
        for new_cfg in sweep_cfg_generator(
            default_cfg_dict,
            tr_param_sweeps=train_param_sweeps,
            ag_param_sweeps=agent_param_sweeps,
            wld_param_sweeps=world_param_sweeps,
            seed_from_timestamp=args.seed_from_timestamp,
            group_name=args.group_name,
        ):
            create_job_dir(
                args.experiment_dir,
                args.job_name_base,
                cfg=new_cfg,
                action_arrays={
                    "consumption_choices": consumption_choices,
                    "work_choices": work_choices,
                    "price_and_wage": price_and_wage,
                    "tax_choices": tax_choices,
                },
            )

    if args.dry_run:
        print("Dry-run -> not actually training...")
    else:
        print("Training multiple experiments locally...")

        # for dirs in experiment dir, run job
        experiment_dirs = [
            f.path for f in os.scandir(args.experiment_dir) if f.is_dir()
        ]
        for experiment in experiment_dirs:
            run_experiment_batch_parallel(
                experiment,
                consumption_choices,
                work_choices,
                price_and_wage,
                tax_choices,
                group_name=args.group_name,
                consumers_only=False,
                no_firms=False,
                default_firm_action=default_firm_action,
                default_government_action=default_government_action,
            )
