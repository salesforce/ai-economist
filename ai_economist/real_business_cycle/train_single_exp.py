# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import argparse

from experiment_utils import run_experiment_batch_parallel
from rbc.constants import all_agents_export_experiment_template

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
    ) = all_agents_export_experiment_template(
        args.num_firms, args.num_consumers, args.num_governments
    )

    if not args.dry_run:
        # for dirs in experiment dir, run job
        experiment = args.experiment_dir
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
