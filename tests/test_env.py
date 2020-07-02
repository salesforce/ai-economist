# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
Unit tests for the wood and stone scenario + basic components
"""

import unittest

from ai_economist import foundation


class CreateEnv:
    """
    Create an environment instance based on a configuration
    """

    def __init__(self):
        self.env = None
        self.set_env_config()

    def set_env_config(self):
        """Set up a sample environment config"""
        self.env_config = {
            # ===== STANDARD ARGUMENTS ======
            "n_agents": 4,  # Number of non-planner agents
            "world_size": [15, 15],  # [Height, Width] of the env world
            "episode_length": 1000,  # Number of time-steps per episode
            # In multi-action-mode, the policy selects an action for each action
            # subspace (defined in component code)
            # Otherwise, the policy selects only 1 action
            "multi_action_mode_agents": False,
            "multi_action_mode_planner": True,
            # When flattening observations, concatenate scalar & vector observations
            # before output
            # Otherwise, return observations with minimal processing
            "flatten_observations": False,
            # When Flattening masks, concatenate each action subspace mask
            # into a single array
            # Note: flatten_masks = True is recommended for masking action logits
            "flatten_masks": True,
            # ===== COMPONENTS =====
            # Which components to use
            "components": [
                # (1) Building houses
                {"Build": {}},
                # (2) Trading collectible resources
                {"ContinuousDoubleAuction": {"max_num_orders": 5}},
                # (3) Movement and resource collection
                {"Gather": {}},
            ],
            # ===== SCENARIO =====
            # Which scenario class to use
            "scenario_name": "uniform/simple_wood_and_stone",
            # (optional) kwargs of the chosen scenario class
            "starting_agent_coin": 10,
            "starting_stone_coverage": 0.10,
            "starting_wood_coverage": 0.10,
        }

        # Create an environment instance from the config
        self.env = foundation.make_env_instance(**self.env_config)


class TestEnv(unittest.TestCase):
    """Unit test to test the env wrapper, reset and step"""

    def test_env_reset_and_step(self):
        """
        Unit tests for the reset and step calls
        """
        create_env = CreateEnv()
        env = create_env.env

        # Assert that the total number of agents matches the sum of the 'n_agents'
        # configuration and the number of planners (1 in this case)
        num_planners = 1
        self.assertEqual(
            len(env.all_agents), create_env.env_config["n_agents"] + num_planners
        )

        # Assert that the number of agents created in the world
        # matches the configuration specification
        self.assertEqual(len(env.world.agents), create_env.env_config["n_agents"])

        # Assert that the planner's index in the world is 'p'
        self.assertEqual(env.world.planner.idx, "p")

        obs = env.reset()

        # Test whether the observation dictionary keys are created as expected
        self.assertEqual(
            sorted(list(obs.keys())),
            [str(i) for i in range(create_env.env_config["n_agents"])] + ["p"],
        )

        obs, reward, done, info = env.step({})

        # Check that the observation, reward and info keys match
        self.assertEqual(obs.keys(), reward.keys())
        self.assertEqual(obs.keys(), info.keys())

        # Assert that __all__ is in done
        assert "__all__" in done


if __name__ == "__main__":
    unittest.main()
