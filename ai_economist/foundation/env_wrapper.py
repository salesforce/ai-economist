# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
The env wrapper class
"""

import GPUtil
import numpy as np
from gym.spaces import Discrete, MultiDiscrete

if len(GPUtil.getAvailable()) > 0:
    from warp_drive.env_wrapper import EnvWrapper
    from warp_drive.utils.recursive_obs_dict_to_spaces_dict import (
        recursive_obs_dict_to_spaces_dict,
    )


class FoundationEnvWrapper(EnvWrapper):
    """
    The environment wrapper class for Foundation.
    This wrapper determines whether the environment reset and steps happen on the
    CPU or the GPU, and proceeds accordingly.
    If the environment runs on the CPU, the reset() and step() calls also occur on
    the CPU.
    If the environment runs on the GPU, only the first reset() happens on the CPU,
    all the relevant data is copied over the GPU after, and the subsequent steps
    all happen on the GPU.
    """

    def __init__(self, *base_args, **base_kwargs):
        super().__init__(*base_args, **base_kwargs)

        # Add observation space for each individual agent to the env
        # when the collated agent "a" is present
        # ----------------------------------------------------------
        obs = self.env.reset()

        obs[str(self.env.n_agents)] = obs["p"]  # planner refers to the final agent id

        if "a" in obs:  # collated agent observation space
            assert isinstance(obs["a"], dict)
            for agent_id in range(self.env.n_agents):
                obs[str(agent_id)] = {}
            for key in obs["a"]:
                assert obs["a"][key].shape[-1] == self.env.n_agents
                for agent_id in range(self.env.n_agents):
                    obs[str(agent_id)][key] = obs["a"][key][..., agent_id]

            self.env.observation_space = recursive_obs_dict_to_spaces_dict(obs)

        # Add action space to the env
        # ---------------------------
        self.env.action_space = {}
        for agent_id in range(len(self.env.world.agents)):
            if self.env.world.agents[agent_id].multi_action_mode:
                self.env.action_space[str([agent_id])] = MultiDiscrete(
                    self.env.get_agent(str(agent_id)).action_spaces
                )
                self.env.action_space[str(agent_id)].dtype = np.int32
                self.env.action_space[
                    str(agent_id)
                ].nvec = self.action_space.nvec.astype(np.int32)

            else:
                self.env.action_space[str(agent_id)] = Discrete(
                    self.env.get_agent(str(agent_id)).action_spaces
                )
                self.env.action_space[str(agent_id)].dtype = np.int32

        if self.env.world.planner.multi_action_mode:
            self.env.action_space["p"] = MultiDiscrete(
                self.env.get_agent("p").action_spaces
            )
            self.env.action_space["p"].dtype = np.int32
            self.env.action_space["p"].nvec = self.action_space_pl.nvec.astype(np.int32)

        else:
            self.env.action_space["p"] = Discrete(self.env.get_agent("p").action_spaces)
            self.env.action_space["p"].dtype = np.int32

        self.env.world.use_cuda = self.use_cuda

        # Steps specific to GPU runs
        # --------------------------
        if self.use_cuda:
            # Register additional cuda functions (other than the scenario step)
            self.reset_on_host = True
            # Component step
            # Create a cuda_component_step dictionary
            self.env.world.cuda_component_step = {}
            for component in self.env.components:
                self.cuda_function_manager.initialize_functions(
                    ["Cuda" + component.name + "Step"]
                )
                self.env.world.cuda_component_step[
                    component.name
                ] = self.cuda_function_manager._get_function(
                    "Cuda" + component.name + "Step"
                )

            # Scenario step and compute reward
            self.cuda_function_manager.initialize_functions(["CudaComputeReward"])
            self.env.cuda_compute_reward = self.cuda_function_manager._get_function(
                "CudaComputeReward"
            )
            # Add to self.env.world for use in components
            self.env.world.cuda_data_manager = self.cuda_data_manager
            self.env.world.cuda_function_manager = self.cuda_function_manager

    def reset_all_envs(self):
        """
        In addition to the reset_all_envs functionality in the base class,
        the data dictionaries for all the component are copied to the device.
        """
        self.env.world.timestep = 0

        if self.reset_on_host:
            # Produce observation
            obs = self.env.reset()
        else:
            assert self.use_cuda

        if self.use_cuda:  # GPU version
            if self.reset_on_host:

                # Helper function to repeat data across the env dimension
                def repeat_across_env_dimension(array, num_envs):
                    return np.stack([array for _ in range(num_envs)], axis=0)

                # Copy host data and tensors to device
                # Note: this happens only once after the first reset on the host

                scenario_and_components = [self.env] + self.env.components

                for item in scenario_and_components:
                    # Add env dimension to data
                    # if "save_copy_and_apply_at_reset" is True
                    data_dictionary = item.get_data_dictionary()
                    tensor_dictionary = item.get_tensor_dictionary()
                    for key in data_dictionary:
                        if data_dictionary[key]["attributes"][
                            "save_copy_and_apply_at_reset"
                        ]:
                            data_dictionary[key]["data"] = repeat_across_env_dimension(
                                data_dictionary[key]["data"], self.n_envs
                            )

                    for key in tensor_dictionary:
                        if tensor_dictionary[key]["attributes"][
                            "save_copy_and_apply_at_reset"
                        ]:
                            tensor_dictionary[key][
                                "data"
                            ] = repeat_across_env_dimension(
                                tensor_dictionary[key]["data"], self.n_envs
                            )

                    self.cuda_data_manager.push_data_to_device(data_dictionary)

                    self.cuda_data_manager.push_data_to_device(
                        tensor_dictionary, torch_accessible=True
                    )

                # All subsequent resets happen on the GPU
                self.reset_on_host = False

                # Return the obs
                return obs
            # Returns an empty dictionary for all subsequent resets on the GPU
            # as arrays are modified in place.

            self.env_resetter.reset_when_done(
                self.cuda_data_manager, mode="force_reset"
            )
            return {}
        return obs  # CPU version

    def step_all_envs(self, actions=None):
        """
        Step through all the environments' components and scenario
        """
        if self.use_cuda:
            # Step through each component
            for component in self.env.components:
                component.component_step()

            # Scenario step
            self.env.scenario_step()

            # Compute rewards
            self.env.generate_rewards()

            obs = {}
            rew = {}
            done = {
                "__all__": self.cuda_data_manager.data_on_device_via_torch("_done_") > 0
            }
            info = {}
        else:
            assert actions is not None
            obs, rew, done, info = self.env.step(actions)

        return obs, rew, done, info
