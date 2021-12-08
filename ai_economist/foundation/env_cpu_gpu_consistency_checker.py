# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
Consistency tests for comparing the cuda (gpu) / no cuda (cpu) version
"""

from collections import defaultdict

import GPUtil

try:
    num_gpus_available = len(GPUtil.getAvailable())
    assert num_gpus_available > 0, "The env consistency checker needs a GPU to run!"
    print(f"{num_gpus_available} GPUs are available.")
    import torch
    from warp_drive.utils.constants import Constants
    from warp_drive.utils.data_feed import DataFeed
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "The env consistency checker requires the 'WarpDrive' package, please run "
        "'pip install rl-warp-drive' first."
    ) from None
except ValueError:
    raise ValueError("The env consistency checker needs a GPU to run!") from None

import numpy as np
from gym.spaces import Discrete, MultiDiscrete

from ai_economist.foundation.env_wrapper import FoundationEnvWrapper

pytorch_cuda_init_success = torch.cuda.FloatTensor(8)
_OBSERVATIONS = Constants.OBSERVATIONS
_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS


def generate_random_actions(env, num_envs):
    """
    Generate random actions for each agent and each env.
    """
    for agent_id in range(env.n_agents + 1):  # Agents + Planner
        if agent_id == env.n_agents:
            agent_id = "p"  # for the planner
        assert isinstance(
            env.action_space[str(agent_id)], (Discrete, MultiDiscrete)
        ), "Unknown action space for env."

    np.random.seed(12)
    action_list = []
    for _ in range(num_envs):
        action_dict = {}
        for agent_id in range(env.n_agents + 1):  # Agents + Planner
            if agent_id == env.n_agents:
                agent_id = "p"  # for the planner
            if isinstance(env.action_space[str(agent_id)], Discrete):
                action_dict[str(agent_id)] = np.random.randint(
                    low=0, high=int(env.action_space[str(agent_id)].n), dtype=np.int32
                )
            else:  # MultiDiscrete action space
                action_dict[str(agent_id)] = np.random.randint(
                    low=0, high=int(env.action_space[str(agent_id)].n), dtype=np.int32
                )
        action_list += [action_dict]
    return action_list


class EnvironmentCPUvsGPU:
    """
    test the rollout consistency between the CPU environment and the GPU environment
    """

    def __init__(
        self,
        env_class,
        env_configs,
        num_envs=2,
        num_episodes=2,
        customized_env_registrar=None,
    ):
        """
        :param env_class: env class to test, for example, TagGridWorld
        :param env_config: env configuration
        :param num_envs: number of parallel example_envs in the test.
        :param num_episodes: number of episodes in the test
            hint: number >=2 is recommended
            since it can fully test the reset
        :param customized_env_registrar: CustomizedEnvironmentRegistrar object
            it provides the customized env info (like src path) for the build

        """
        self.env_class = env_class
        self.env_configs = env_configs
        self.num_envs = num_envs
        self.num_episodes = num_episodes
        self.customized_env_registrar = customized_env_registrar

    def test_env_reset_and_step(self, consistency_threshold_pct=1):
        """
        Perform consistency checks for the reset() and step() functions
        consistency_threshold_pct: consistency threshold as a percentage
        (defaults to 1%).
        """

        for scenario in self.env_configs:
            env_config = self.env_configs[scenario]

            print(f"Running {scenario}...")

            # Env Reset
            # CPU version of env
            env_cpu = {}
            obs_dict_of_lists = defaultdict(list)
            obs_cpu = {}

            for env_id in range(self.num_envs):
                env_cpu[env_id] = self.env_class(**env_config)
                # Set use_cuda to False for the CPU envs
                env_cpu[env_id].use_cuda = False
                env_cpu[env_id].world.use_cuda = False

                # obs will be a nested dict of
                # {policy_id: combined_agent_obs for each subkey}
                obs = env_cpu[env_id].reset()

                for key in obs:
                    assert isinstance(obs[key], dict)
                    for subkey in obs[key]:
                        obs_dict_of_lists[_OBSERVATIONS + "_" + key + "_" + subkey] += [
                            obs[key][subkey]
                        ]

            for key in obs_dict_of_lists:
                obs_cpu[key] = np.stack((obs_dict_of_lists[key]), axis=0)

            # GPU version of env
            env_gpu = FoundationEnvWrapper(
                self.env_class(**env_config),
                num_envs=self.num_envs,
                use_cuda=True,
                customized_env_registrar=self.customized_env_registrar,
            )
            env_gpu.reset_all_envs()

            # Observations placeholders
            # -------------------------
            data_feed = DataFeed()
            for key in obs_cpu:
                data_feed.add_data(
                    name=key,
                    data=obs_cpu[key],
                    save_copy_and_apply_at_reset=True,
                )

            # Define a mapping function: policy -> agent_ids
            # ----------------------------------------------
            policy_to_agent_ids_mapping = {
                "a": [str(agent_id) for agent_id in range(env_config["n_agents"])],
                "p": ["p"],
            }

            for key in policy_to_agent_ids_mapping:
                if len(policy_to_agent_ids_mapping[key]) == 1:
                    rewards_data = np.zeros(
                        (self.num_envs,),
                        dtype=np.float32,
                    )
                else:
                    rewards_data = np.zeros(
                        (self.num_envs, len(policy_to_agent_ids_mapping[key])),
                        dtype=np.float32,
                    )
                data_feed.add_data(
                    name=_REWARDS + "_" + key,
                    data=rewards_data,
                    save_copy_and_apply_at_reset=True,
                )

            env_gpu.cuda_data_manager.push_data_to_device(
                data_feed, torch_accessible=True
            )

            # Consistency checks at the first reset
            # -------------------------------------
            obs_gpu = {}
            print("Running obs consistency check after the first reset.")
            for key in obs_cpu:
                obs_gpu[key] = env_gpu.cuda_data_manager.pull_data_from_device(key)
                self.run_consistency_checks(
                    obs_cpu[key], obs_gpu[key], threshold_pct=consistency_threshold_pct
                )

            # Consistency checks during step
            # ------------------------------
            print("Running obs/rew/done consistency check during env resets and steps")

            # Test across multiple episodes
            for _ in range(self.num_episodes * env_config["episode_length"]):
                actions_list_of_dicts = generate_random_actions(
                    env_gpu.env, self.num_envs
                )

                for policy in policy_to_agent_ids_mapping:
                    actions_list = []
                    for action_dict in actions_list_of_dicts:
                        combined_actions = np.stack(
                            [
                                action_dict[key]
                                for key in policy_to_agent_ids_mapping[policy]
                            ],
                            axis=0,
                        )
                        actions_list += [combined_actions]

                    actions = np.stack((actions_list), axis=0)
                    name = _ACTIONS + "_" + policy
                    actions_data = DataFeed()
                    actions_data.add_data(name=name, data=actions)

                    if not env_gpu.cuda_data_manager.is_data_on_device_via_torch(name):
                        env_gpu.cuda_data_manager.push_data_to_device(
                            actions_data, torch_accessible=True
                        )
                    else:
                        env_gpu.cuda_data_manager.data_on_device_via_torch(name)[
                            :
                        ] = torch.from_numpy(actions)

                obs_cpu = {}
                obs_dict_of_lists = defaultdict(list)
                rew_dict_of_lists = defaultdict(list)
                rew_cpu = {}
                done_list = []

                for env_id in range(self.num_envs):
                    obs, rew, done, _ = env_cpu[env_id].step(
                        actions_list_of_dicts[env_id]
                    )

                    for key in obs:
                        assert isinstance(obs[key], dict)
                        for subkey in obs[key]:
                            obs_dict_of_lists[
                                _OBSERVATIONS + "_" + key + "_" + subkey
                            ] += [obs[key][subkey]]

                    for key in rew:
                        rew_dict_of_lists[_REWARDS + "_" + key] += [rew[key]]

                    done_list += [done]

                for key in obs_dict_of_lists:
                    obs_cpu[key] = np.stack((obs_dict_of_lists[key]), axis=0)

                for key in rew_dict_of_lists:
                    rew_cpu[key] = np.stack((rew_dict_of_lists[key]), axis=0)

                done_cpu = {
                    "__all__": np.array([done["__all__"] for done in done_list])
                }

                # Step through all the environments
                env_gpu.step_all_envs()

                obs_gpu = {}
                for key in obs_cpu:
                    obs_gpu[key] = env_gpu.cuda_data_manager.pull_data_from_device(key)
                    self.run_consistency_checks(
                        obs_cpu[key],
                        obs_gpu[key],
                        threshold_pct=consistency_threshold_pct,
                    )

                rew_gpu = {}
                for key in rew_cpu:
                    rew_gpu[key] = env_gpu.cuda_data_manager.pull_data_from_device(key)
                    self.run_consistency_checks(
                        rew_cpu[key],
                        rew_gpu[key],
                        threshold_pct=consistency_threshold_pct,
                    )

                done_gpu = (
                    env_gpu.cuda_data_manager.data_on_device_via_torch("_done_")
                    .cpu()
                    .numpy()
                )
                assert all(done_cpu["__all__"] == (done_gpu > 0))

                # GPU reset
                env_gpu.reset_only_done_envs()

                # Now, pull done flags and they should be set to 0 (False) again
                done_gpu = env_gpu.cuda_data_manager.pull_data_from_device("_done_")
                assert done_gpu.sum() == 0

                # CPU reset
                for env_id in range(self.num_envs):
                    if done_cpu["__all__"][env_id]:
                        # Reset the CPU for this env_id
                        obs = env_cpu[env_id].reset()

                        obs_cpu = {}
                        obs_dict_of_lists = defaultdict(list)
                        for key in obs:
                            assert isinstance(obs[key], dict)
                            for subkey in obs[key]:
                                obs_dict_of_lists[
                                    _OBSERVATIONS + "_" + key + "_" + subkey
                                ] += [obs[key][subkey]]

                        for key in obs_dict_of_lists:
                            obs_cpu[key] = np.stack((obs_dict_of_lists[key]), axis=0)

                        obs_gpu = {}
                        for key in obs_cpu:
                            obs_gpu[
                                key
                            ] = env_gpu.cuda_data_manager.pull_data_from_device(key)
                            self.run_consistency_checks(
                                obs_cpu[key],
                                obs_gpu[key][env_id],
                                threshold_pct=consistency_threshold_pct,
                            )
            print(
                f"The CPU and the GPU environment outputs are consistent "
                f"within {consistency_threshold_pct} percent."
            )

    @staticmethod
    def run_consistency_checks(cpu_value, gpu_value, threshold_pct=1):
        """
        Perform consistency checks between the cpu and gpu values.
        The default threshold is 2 decimal places (1 %).
        """
        epsilon = 1e-10  # a small number for preventing indeterminate divisions
        max_abs_diff = np.max(np.abs(cpu_value - gpu_value))
        relative_max_abs_diff_pct = (
            np.max(np.abs((cpu_value - gpu_value) / (epsilon + cpu_value))) * 100.0
        )
        # Assert that the max absolute difference is smaller than the threshold
        # or the relative_max_abs_diff_pct is smaller (when the values are high)
        assert (
            max_abs_diff < threshold_pct / 100.0
            or relative_max_abs_diff_pct < threshold_pct
        )
