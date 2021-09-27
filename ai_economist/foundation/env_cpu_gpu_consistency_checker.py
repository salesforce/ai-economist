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
import numpy as np
from gym.spaces import Discrete, MultiDiscrete

from ai_economist.foundation.env_wrapper import FoundationEnvWrapper

if len(GPUtil.getAvailable()) > 0:
    import torch
    from warp_drive.utils.constants import Constants
    from warp_drive.utils.data_feed import DataFeed

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
        use_gpu_testing_mode=True,
        customized_env_registrar=None,
    ):
        """
        :param env_class: env class to test, for example, TagGridWorld
        :param env_config: env configuration
        :param num_envs: number of parallel example_envs in the test.
            If use_gpu_testing_mode = True,
            num_envs = 2 and num_agents=5 are enforced
        :param num_episodes: number of episodes in the test
            hint: number >=2 is recommended
            since it can fully test the reset
        :param use_gpu_testing_mode: determine whether to simply load the
            discrete_and_continuous_tag_envs.cubin or compile the .cu source
            code to create a .cubin.
            If use_gpu_testing_mode = True: do not forget to
            include your testing env into discrete_and_continuous_tag_envs.cu
            and build it. This is the recommended flow because the
            Makefile will automate this build.
        :param customized_env_registrar: CustomizedEnvironmentRegistrar object
            it provides the customized env info (like src path) for the build

        """
        self.env_class = env_class
        self.env_configs = env_configs
        if use_gpu_testing_mode:
            print(
                f"enforce the num_envs = {num_envs} because you have "
                f"use_gpu_testing_mode = True, where the cubin file"
                f"supporting this testing mode assumes 2 parallel example_envs"
            )
        self.num_envs = num_envs
        self.num_episodes = num_episodes
        self.use_gpu_testing_mode = use_gpu_testing_mode
        self.customized_env_registrar = customized_env_registrar

    def test_env_reset_and_step(self):
        """
        Perform consistency checks for the reset() and step() functions
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
                testing_mode=self.use_gpu_testing_mode,
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
                self.run_consistency_checks(obs_cpu[key], obs_gpu[key])

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

                # Update actions tensor on the gpu
                _, _, done_gpu, _ = env_gpu.step()
                done_gpu["__all__"] = done_gpu["__all__"].cpu().numpy()

                obs_gpu = {}
                for key in obs_cpu:
                    obs_gpu[key] = env_gpu.cuda_data_manager.pull_data_from_device(key)
                    self.run_consistency_checks(obs_cpu[key], obs_gpu[key])

                rew_gpu = {}
                for key in rew_cpu:
                    rew_gpu[key] = env_gpu.cuda_data_manager.pull_data_from_device(key)
                    self.run_consistency_checks(rew_cpu[key], rew_gpu[key])

                assert all(done_cpu["__all__"] == done_gpu["__all__"])

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
                                obs_cpu[key], obs_gpu[key][env_id]
                            )

    @staticmethod
    def run_consistency_checks(cpu_value, gpu_value, decimal_places=3):
        """
        Perform consistency checks between the cpu and gpu values.
        The default threshold is 3 decimal places.
        """
        assert np.max(np.abs(cpu_value - gpu_value)) < 10 ** (-decimal_places)
