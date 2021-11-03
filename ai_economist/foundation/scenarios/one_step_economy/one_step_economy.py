# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from ai_economist.foundation.base.base_env import BaseEnvironment, scenario_registry
from ai_economist.foundation.scenarios.utils import rewards, social_metrics


@scenario_registry.add
class OneStepEconomy(BaseEnvironment):
    """
    A simple model featuring one "step" of setting taxes and earning income.

    As described in https://arxiv.org/abs/2108.02755:
        A simplified version of simple_wood_and_stone scenario where both the planner
        and the agents each make a single decision: the planner setting taxes and the
        agents choosing labor. Each agent chooses an amount of labor that optimizes
        its post-tax utility, and this optimal labor depends on its skill and the tax
        rates, and it does not depend on the labor choices of other agents. Before
        the agents act, the planner sets the marginal tax rates in order to optimize
        social welfare.

    Note:
        This scenario is intended to be used with the 'PeriodicBracketTax' and
            'SimpleLabor' components.
        It should use an episode length of 2. In the first step, taxes are set by
            the planner via 'PeriodicBracketTax'. In the second, agents select how much
            to work/earn via 'SimpleLabor'.

    Args:
        agent_reward_type (str): The type of utility function used to compute each
            agent's reward. Defaults to "coin_minus_labor_cost".
        isoelastic_eta (float): The shape parameter of the isoelastic function used
            in the "isoelastic_coin_minus_labor" utility function.
        labor_exponent (float): The labor exponent parameter used in the
            "coin_minus_labor_cost" utility function.
        labor_cost (float): The coefficient used to weight the cost of labor.
        planner_reward_type (str): The type of social welfare function (SWF) used to
            compute the planner's reward. Defaults to "inv_income_weighted_utility".
        mixing_weight_gini_vs_coin (float): Must be between 0 and 1 (inclusive).
            Controls the weighting of equality and productivity when using SWF
            "coin_eq_times_productivity", where a value of 0 (default) yields equal
            weighting, and 1 only considers productivity.
    """

    name = "one-step-economy"
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]
    required_entities = ["Coin"]

    def __init__(
        self,
        *base_env_args,
        agent_reward_type="coin_minus_labor_cost",
        isoelastic_eta=0.23,
        labor_exponent=2.0,
        labor_cost=1.0,
        planner_reward_type="inv_income_weighted_utility",
        mixing_weight_gini_vs_coin=0,
        **base_env_kwargs
    ):
        super().__init__(*base_env_args, **base_env_kwargs)

        self.num_agents = len(self.world.agents)

        self.labor_cost = labor_cost
        self.agent_reward_type = agent_reward_type
        self.isoelastic_eta = isoelastic_eta
        self.labor_exponent = labor_exponent
        self.planner_reward_type = planner_reward_type
        self.mixing_weight_gini_vs_coin = mixing_weight_gini_vs_coin
        self.planner_starting_coin = 0

        self.curr_optimization_metrics = {str(a.idx): 0 for a in self.all_agents}

    # The following methods must be implemented for each scenario
    # -----------------------------------------------------------
    def reset_starting_layout(self):
        """
        Part 1/2 of scenario reset. This method handles resetting the state of the
        environment managed by the scenario (i.e. resource & landmark layout).

        Here, generate a resource source layout consistent with target parameters.
        """

    def reset_agent_states(self):
        """
        Part 2/2 of scenario reset. This method handles resetting the state of the
        agents themselves (i.e. inventory, locations, etc.).

        Here, empty inventories, give mobile agents any starting coin, and place them
        in random accesible locations to start.
        """
        self.world.clear_agent_locs()

        for agent in self.world.agents:
            # Clear everything to start with
            agent.state["inventory"] = {k: 0 for k in agent.state["inventory"].keys()}
            agent.state["escrow"] = {k: 0 for k in agent.state["escrow"].keys()}
            agent.state["endogenous"] = {k: 0 for k in agent.state["endogenous"].keys()}

        self.world.planner.inventory["Coin"] = self.planner_starting_coin

    def scenario_step(self):
        """
        Update the state of the world according to whatever rules this scenario
        implements.

        This gets called in the 'step' method (of base_env) after going through each
        component step and before generating observations, rewards, etc.

        NOTE: does not take agent actions into account.
        """

    def generate_observations(self):
        """
        Generate observations associated with this scenario.

        A scenario does not need to produce observations and can provide observations
        for only some agent types; however, for a given agent type, it should either
        always or never yield an observation. If it does yield an observation,
        that observation should always have the same structure/sizes!

        Returns:
            obs (dict): A dictionary of {agent.idx: agent_obs_dict}. In words,
                return a dictionary with an entry for each agent (which can including
                the planner) for which this scenario provides an observation. For each
                entry, the key specifies the index of the agent and the value contains
                its associated observation dictionary.

        Here, non-planner agents receive spatial observations (depending on the env
        config) as well as the contents of their inventory and endogenous quantities.
        The planner also receives spatial observations (again, depending on the env
        config) as well as the inventory of each of the mobile agents.
        """
        obs_dict = dict()
        for agent in self.world.agents:
            obs_dict[str(agent.idx)] = {}

        coin_endowments = np.array(
            [agent.total_endowment("Coin") for agent in self.world.agents]
        )
        equality = social_metrics.get_equality(coin_endowments)
        productivity = social_metrics.get_productivity(coin_endowments)
        normalized_per_capita_productivity = productivity / self.num_agents / 1000
        obs_dict[self.world.planner.idx] = {
            "normalized_per_capita_productivity": normalized_per_capita_productivity,
            "equality": equality,
        }

        return obs_dict

    def compute_reward(self):
        """
        Apply the reward function(s) associated with this scenario to get the rewards
        from this step.

        Returns:
            rew (dict): A dictionary of {agent.idx: agent_obs_dict}. In words,
                return a dictionary with an entry for each agent in the environment
                (including the planner). For each entry, the key specifies the index of
                the agent and the value contains the scalar reward earned this timestep.

        Rewards are computed as the marginal utility (agents) or marginal social
        welfare (planner) experienced on this timestep. Ignoring discounting,
        this means that agents' (planner's) objective is to maximize the utility
        (social welfare) associated with the terminal state of the episode.
        """
        curr_optimization_metrics = self.get_current_optimization_metrics(
            self.world.agents,
            isoelastic_eta=float(self.isoelastic_eta),
            labor_exponent=float(self.labor_exponent),
            labor_coefficient=float(self.labor_cost),
        )
        planner_agents_rew = {
            k: v - self.curr_optimization_metrics[k]
            for k, v in curr_optimization_metrics.items()
        }
        self.curr_optimization_metrics = curr_optimization_metrics
        return planner_agents_rew

    # Optional methods for customization
    # ----------------------------------
    def additional_reset_steps(self):
        """
        Extra scenario-specific steps that should be performed at the end of the reset
        cycle.

        For each reset cycle...
            First, reset_starting_layout() and reset_agent_states() will be called.

            Second, <component>.reset() will be called for each registered component.

            Lastly, this method will be called to allow for any final customization of
            the reset cycle.
        """
        self.curr_optimization_metrics = self.get_current_optimization_metrics(
            self.world.agents,
            isoelastic_eta=float(self.isoelastic_eta),
            labor_exponent=float(self.labor_exponent),
            labor_coefficient=float(self.labor_cost),
        )

    def scenario_metrics(self):
        """
        Allows the scenario to generate metrics (collected along with component metrics
        in the 'metrics' property).

        To have the scenario add metrics, this function needs to return a dictionary of
        {metric_key: value} where 'value' is a scalar (no nesting or lists!)

        Here, summarize social metrics, endowments, utilities, and labor cost annealing.
        """
        metrics = dict()

        # Log social/economic indicators
        coin_endowments = np.array(
            [agent.total_endowment("Coin") for agent in self.world.agents]
        )
        pretax_incomes = np.array(
            [agent.state["production"] for agent in self.world.agents]
        )
        metrics["social/productivity"] = social_metrics.get_productivity(
            coin_endowments
        )
        metrics["social/equality"] = social_metrics.get_equality(coin_endowments)

        utilities = np.array(
            [self.curr_optimization_metrics[agent.idx] for agent in self.world.agents]
        )
        metrics[
            "social_welfare/coin_eq_times_productivity"
        ] = rewards.coin_eq_times_productivity(
            coin_endowments=coin_endowments, equality_weight=1.0
        )
        metrics[
            "social_welfare/inv_income_weighted_utility"
        ] = rewards.inv_income_weighted_utility(
            coin_endowments=pretax_incomes, utilities=utilities  # coin_endowments,
        )

        # Log average endowments, endogenous, and utility for agents
        agent_endows = {}
        agent_endogenous = {}
        agent_utilities = []
        for agent in self.world.agents:
            for resource in agent.inventory.keys():
                if resource not in agent_endows:
                    agent_endows[resource] = []
                agent_endows[resource].append(
                    agent.inventory[resource] + agent.escrow[resource]
                )

            for endogenous, quantity in agent.endogenous.items():
                if endogenous not in agent_endogenous:
                    agent_endogenous[endogenous] = []
                agent_endogenous[endogenous].append(quantity)

            agent_utilities.append(self.curr_optimization_metrics[agent.idx])

        for resource, quantities in agent_endows.items():
            metrics["endow/avg_agent/{}".format(resource)] = np.mean(quantities)

        for endogenous, quantities in agent_endogenous.items():
            metrics["endogenous/avg_agent/{}".format(endogenous)] = np.mean(quantities)

        metrics["util/avg_agent"] = np.mean(agent_utilities)

        # Log endowments and utility for the planner
        for resource, quantity in self.world.planner.inventory.items():
            metrics["endow/p/{}".format(resource)] = quantity

        metrics["util/p"] = self.curr_optimization_metrics[self.world.planner.idx]

        return metrics

    def get_current_optimization_metrics(
        self, agents, isoelastic_eta=0.23, labor_exponent=2.0, labor_coefficient=0.1
    ):
        """
        Compute optimization metrics based on the current state. Used to compute reward.

        Returns:
            curr_optimization_metric (dict): A dictionary of {agent.idx: metric}
                with an entry for each agent (including the planner) in the env.
        """
        curr_optimization_metric = {}

        coin_endowments = np.array([agent.total_endowment("Coin") for agent in agents])

        pretax_incomes = np.array([agent.state["production"] for agent in agents])

        # Optimization metric for agents:
        for agent in agents:
            if self.agent_reward_type == "isoelastic_coin_minus_labor":
                assert 0.0 <= isoelastic_eta <= 1.0
                curr_optimization_metric[
                    agent.idx
                ] = rewards.isoelastic_coin_minus_labor(
                    coin_endowment=agent.total_endowment("Coin"),
                    total_labor=agent.state["endogenous"]["Labor"],
                    isoelastic_eta=isoelastic_eta,
                    labor_coefficient=labor_coefficient,
                )
            elif self.agent_reward_type == "coin_minus_labor_cost":
                assert labor_exponent > 1.0
                curr_optimization_metric[agent.idx] = rewards.coin_minus_labor_cost(
                    coin_endowment=agent.total_endowment("Coin"),
                    total_labor=agent.state["endogenous"]["Labor"],
                    labor_exponent=labor_exponent,
                    labor_coefficient=labor_coefficient,
                )
        # Optimization metric for the planner:
        if self.planner_reward_type == "coin_eq_times_productivity":
            curr_optimization_metric[
                self.world.planner.idx
            ] = rewards.coin_eq_times_productivity(
                coin_endowments=coin_endowments,
                equality_weight=1 - self.mixing_weight_gini_vs_coin,
            )
        elif self.planner_reward_type == "inv_income_weighted_utility":
            curr_optimization_metric[
                self.world.planner.idx
            ] = rewards.inv_income_weighted_utility(
                coin_endowments=pretax_incomes,  # coin_endowments,
                utilities=np.array(
                    [curr_optimization_metric[agent.idx] for agent in agents]
                ),
            )
        else:
            print("No valid planner reward selected!")
            raise NotImplementedError
        return curr_optimization_metric
