# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
The env wrapper class
"""

import logging

import GPUtil

try:
    num_gpus_available = len(GPUtil.getAvailable())
    print(f"Inside env_wrapper.py: {num_gpus_available} GPUs are available.")
    if num_gpus_available == 0:
        print("No GPUs found! Running the simulation on a CPU.")
    else:
        from warp_drive.managers.data_manager import CUDADataManager
        from warp_drive.managers.function_manager import (
            CUDAEnvironmentReset,
            CUDAFunctionManager,
        )
except ModuleNotFoundError:
    print(
        "Warning: The 'WarpDrive' package is not found and cannot be used! "
        "If you wish to use WarpDrive, please run "
        "'pip install rl-warp-drive' first."
    )
except ValueError:
    print("No GPUs found! Running the simulation on a CPU.")

import numpy as np
from gym.spaces import Box, Dict, Discrete, MultiDiscrete

BIG_NUMBER = 1e20


def recursive_obs_dict_to_spaces_dict(obs):
    """Recursively return the observation space dictionary
    for a dictionary of observations

    Args:
        obs (dict): A dictionary of observations keyed by agent index
        for a multi-agent environment

    Returns:
        Dict: A dictionary (space.Dict) of observation spaces
    """
    assert isinstance(obs, dict)
    dict_of_spaces = {}
    for k, v in obs.items():

        # list of lists are listified np arrays
        _v = v
        if isinstance(v, list):
            _v = np.array(v)
        elif isinstance(v, (int, np.integer, float, np.floating)):
            _v = np.array([v])

        # assign Space
        if isinstance(_v, np.ndarray):
            x = float(BIG_NUMBER)
            box = Box(low=-x, high=x, shape=_v.shape, dtype=_v.dtype)
            low_high_valid = (box.low < 0).all() and (box.high > 0).all()

            # This loop avoids issues with overflow to make sure low/high are good.
            while not low_high_valid:
                x = x // 2
                box = Box(low=-x, high=x, shape=_v.shape, dtype=_v.dtype)
                low_high_valid = (box.low < 0).all() and (box.high > 0).all()

            dict_of_spaces[k] = box

        elif isinstance(_v, dict):
            dict_of_spaces[k] = recursive_obs_dict_to_spaces_dict(_v)
        else:
            raise TypeError
    return Dict(dict_of_spaces)


class FoundationEnvWrapper:
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

    def __init__(
        self,
        env_obj=None,
        env_name=None,
        env_config=None,
        num_envs=1,
        use_cuda=False,
        env_registrar=None,
        event_messenger=None,
        process_id=0,
    ):
        """
        'env_obj': an environment object
        'env_name': an environment name that is registered on the
            WarpDrive environment registrar
        'env_config': environment configuration to instantiate
            an environment from the registrar
        'use_cuda': if True, step through the environment on the GPU, else on the CPU
        'num_envs': the number of parallel environments to instantiate. Note: this is
            only relevant when use_cuda is True
        'env_registrar': EnvironmentRegistrar object
            it provides the customized env info (like src path) for the build
        'event_messenger': multiprocessing Event to sync up the build
            when using multiple processes
        'process_id': id of the process running WarpDrive
        """
        # Need to pass in an environment instance
        if env_obj is not None:
            self.env = env_obj
        else:
            assert (
                env_name is not None
                and env_config is not None
                and env_registrar is not None
            )
            self.env = env_registrar.get(env_name, use_cuda)(**env_config)

        self.n_agents = self.env.num_agents
        self.episode_length = self.env.episode_length

        assert self.env.name
        self.name = self.env.name

        # Add observation space to the env
        # --------------------------------
        # Note: when the collated agent "a" is present, add obs keys
        # for each individual agent to the env
        # and remove the collated agent "a" from the observation
        obs = self.obs_at_reset()
        self.env.observation_space = recursive_obs_dict_to_spaces_dict(obs)

        # Add action space to the env
        # ---------------------------
        self.env.action_space = {}
        for agent_id in range(len(self.env.world.agents)):
            if self.env.world.agents[agent_id].multi_action_mode:
                self.env.action_space[str([agent_id])] = MultiDiscrete(
                    self.env.get_agent(str(agent_id)).action_spaces
                )
            else:
                self.env.action_space[str(agent_id)] = Discrete(
                    self.env.get_agent(str(agent_id)).action_spaces
                )
            self.env.action_space[str(agent_id)].dtype = np.int32

        if self.env.world.planner.multi_action_mode:
            self.env.action_space["p"] = MultiDiscrete(
                self.env.get_agent("p").action_spaces
            )
        else:
            self.env.action_space["p"] = Discrete(self.env.get_agent("p").action_spaces)
        self.env.action_space["p"].dtype = np.int32

        # Ensure the observation and action spaces share the same keys
        assert set(self.env.observation_space.keys()) == set(
            self.env.action_space.keys()
        )

        # CUDA-specific initializations
        # -----------------------------
        # Flag to determine whether to use CUDA or not
        self.use_cuda = use_cuda
        if self.use_cuda:
            assert len(GPUtil.getAvailable()) > 0, (
                "The env wrapper needs a GPU to run" " when use_cuda is True!"
            )
            assert hasattr(self.env, "use_cuda")
            assert hasattr(self.env, "cuda_data_manager")
            assert hasattr(self.env, "cuda_function_manager")

            assert hasattr(self.env.world, "use_cuda")
            assert hasattr(self.env.world, "cuda_data_manager")
            assert hasattr(self.env.world, "cuda_function_manager")
        self.env.use_cuda = use_cuda
        self.env.world.use_cuda = self.use_cuda

        # Flag to determine where the reset happens (host or device)
        # First reset is always on the host (CPU), and subsequent resets are on
        # the device (GPU)
        self.reset_on_host = True

        # Steps specific to GPU runs
        # --------------------------
        if self.use_cuda:
            logging.info("USING CUDA...")

            # Number of environments to run in parallel
            assert num_envs >= 1
            self.n_envs = num_envs

            logging.info("Initializing the CUDA data manager...")
            self.cuda_data_manager = CUDADataManager(
                num_agents=self.n_agents,
                episode_length=self.episode_length,
                num_envs=self.n_envs,
            )

            logging.info("Initializing the CUDA function manager...")
            self.cuda_function_manager = CUDAFunctionManager(
                num_agents=int(self.cuda_data_manager.meta_info("n_agents")),
                num_envs=int(self.cuda_data_manager.meta_info("n_envs")),
                process_id=process_id,
            )
            self.cuda_function_manager.compile_and_load_cuda(
                env_name=self.name,
                template_header_file="template_env_config.h",
                template_runner_file="template_env_runner.cu",
                customized_env_registrar=env_registrar,
                event_messenger=event_messenger,
            )

            # Register the CUDA step() function for the env
            # Note: generate_observation() and compute_reward()
            # should be part of the step function itself
            step_function = f"Cuda{self.name}Step"
            self.cuda_function_manager.initialize_functions([step_function])
            self.env.cuda_step = self.cuda_function_manager.get_function(step_function)

            # Register additional cuda functions (other than the scenario step)
            # Component step
            # Create a cuda_component_step dictionary
            self.env.world.cuda_component_step = {}
            for component in self.env.components:
                self.cuda_function_manager.initialize_functions(
                    ["Cuda" + component.name + "Step"]
                )
                self.env.world.cuda_component_step[
                    component.name
                ] = self.cuda_function_manager.get_function(
                    "Cuda" + component.name + "Step"
                )

            # Compute reward
            self.cuda_function_manager.initialize_functions(["CudaComputeReward"])
            self.env.cuda_compute_reward = self.cuda_function_manager.get_function(
                "CudaComputeReward"
            )

            # Add wrapper attributes for use within env
            self.env.cuda_data_manager = self.cuda_data_manager
            self.env.cuda_function_manager = self.cuda_function_manager

            # Register the env resetter
            self.env_resetter = CUDAEnvironmentReset(
                function_manager=self.cuda_function_manager
            )

            # Add to self.env.world for use in components
            self.env.world.cuda_data_manager = self.cuda_data_manager
            self.env.world.cuda_function_manager = self.cuda_function_manager

    def reset_all_envs(self):
        """
        Reset the state of the environment to initialize a new episode.
        if self.reset_on_host is True:
            calls the CPU env to prepare and return the initial state
        if self.use_cuda is True:
            if self.reset_on_host is True:
                expands initial state to parallel example_envs and push to GPU once
                sets self.reset_on_host = False
            else:
                calls device hard reset managed by the CUDAResetter
        """
        self.env.world.timestep = 0

        if self.reset_on_host:
            # Produce observation
            obs = self.obs_at_reset()
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
            # as arrays are modified in place
            self.env_resetter.reset_when_done(
                self.cuda_data_manager, mode="force_reset"
            )
            return {}
        return obs  # CPU version

    def reset_only_done_envs(self):
        """
        This function only works for GPU example_envs.
        It will check all the running example_envs,
        and only resets those example_envs that are observing done flag is True
        """
        assert self.use_cuda and not self.reset_on_host, (
            "reset_only_done_envs() only works "
            "for self.use_cuda = True and self.reset_on_host = False"
        )

        self.env_resetter.reset_when_done(self.cuda_data_manager, mode="if_done")
        return {}

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

            result = None  # Do not return anything
        else:
            assert actions is not None, "Please provide actions to step with."
            obs, rew, done, info = self.env.step(actions)
            obs = self._reformat_obs(obs)
            rew = self._reformat_rew(rew)
            result = obs, rew, done, info
        return result

    def obs_at_reset(self):
        """
        Calls the (Python) env to reset and return the initial state
        """
        obs = self.env.reset()
        obs = self._reformat_obs(obs)
        return obs

    def _reformat_obs(self, obs):
        if "a" in obs:
            # This means the env uses collated obs.
            # Set each individual agent as obs keys for processing with WarpDrive.
            for agent_id in range(self.env.n_agents):
                obs[str(agent_id)] = {}
                for key in obs["a"].keys():
                    obs[str(agent_id)][key] = obs["a"][key][..., agent_id]
            del obs["a"]  # remove the key "a"
        return obs

    def _reformat_rew(self, rew):
        if "a" in rew:
            # This means the env uses collated rew.
            # Set each individual agent as rew keys for processing with WarpDrive.
            assert isinstance(rew, dict)
            for agent_id in range(self.env.n_agents):
                rew[str(agent_id)] = rew["a"][agent_id]
            del rew["a"]  # remove the key "a"
        return rew

    def reset(self):
        """
        Alias for reset_all_envs() when CPU is used (conforms to gym-style)
        """
        return self.reset_all_envs()

    def step(self, actions=None):
        """
        Alias for step_all_envs() when CPU is used (conforms to gym-style)
        """
        return self.step_all_envs(actions)
