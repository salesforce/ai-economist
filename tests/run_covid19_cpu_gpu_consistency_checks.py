# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import os

from warp_drive.utils.env_registrar import CustomizedEnvironmentRegistrar

from ai_economist.foundation.env_cpu_gpu_consistency_checker import EnvironmentCPUvsGPU
from ai_economist.foundation.scenarios.covid19.covid19_env import (
    CovidAndEconomyEnvironment,
)

env_registrar = CustomizedEnvironmentRegistrar()
this_file_dir = os.path.dirname(os.path.abspath(__file__))
env_registrar.register_environment(
    CovidAndEconomyEnvironment.name,
    os.path.join(this_file_dir, "../ai_economist/foundation/scenarios/covid19/covid19_build.cu")
)
env_configs = {
    "test1": {
        "collate_agent_step_and_reset_data": True,
        "components": [
            {"ControlUSStateOpenCloseStatus": {"action_cooldown_period": 28}},
            {
                "FederalGovernmentSubsidy": {
                    "num_subsidy_levels": 20,
                    "subsidy_interval": 90,
                    "max_annual_subsidy_per_person": 20000,
                }
            },
            {
                "VaccinationCampaign": {
                    "daily_vaccines_per_million_people": 3000,
                    "delivery_interval": 1,
                    "vaccine_delivery_start_date": "2021-01-12",
                }
            },
        ],
        "economic_reward_crra_eta": 2,
        "episode_length": 540,
        "flatten_masks": True,
        "flatten_observations": False,
        "health_priority_scaling_agents": 0.3,
        "health_priority_scaling_planner": 0.45,
        "infection_too_sick_to_work_rate": 0.1,
        "multi_action_mode_agents": False,
        "multi_action_mode_planner": False,
        "n_agents": 51,
        "path_to_data_and_fitted_params": "",
        "pop_between_age_18_65": 0.6,
        "risk_free_interest_rate": 0.03,
        "world_size": [1, 1],
        "start_date": "2020-03-22",
        "use_real_world_data": False,
        "use_real_world_policies": False,
    }
}

testing_class = EnvironmentCPUvsGPU(
    env_class=CovidAndEconomyEnvironment,
    env_configs=env_configs,
    num_envs=2,
    num_episodes=2,
    use_gpu_testing_mode=False,
    customized_env_registrar=env_registrar,
)

testing_class.test_env_reset_and_step()
