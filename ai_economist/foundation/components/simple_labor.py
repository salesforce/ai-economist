# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from ai_economist.foundation.base.base_component import (
    BaseComponent,
    component_registry,
)


@component_registry.add
class SimpleLabor(BaseComponent):
    """
    Allows Agents to select a level of labor, which earns income based on skill.

    Labor is "simple" because this simplifies labor to a choice along a 1D axis. More
    concretely, this component adds 100 labor actions, each representing a choice of
    how many hours to work, e.g. action 50 represents doing 50 hours of work; each
    Agent earns income proportional to the product of its labor amount (representing
    hours worked) and its skill (representing wage), with higher skill and higher labor
    yielding higher income.

    This component is intended to be used with the 'PeriodicBracketTax' component and
    the 'one-step-economy' scenario.

    Args:
        mask_first_step (bool): Defaults to True. If True, masks all non-0 labor
            actions on the first step of the environment. When combined with the
            intended component/scenario, the first env step is used to set taxes
            (via the 'redistribution' component) and the second step is used to
            select labor (via this component).
        payment_max_skill_multiplier (float): When determining the skill level of
            each Agent, sampled skills are clipped to this maximum value.
    """

    name = "SimpleLabor"
    required_entities = ["Coin"]
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
        self,
        *base_component_args,
        mask_first_step=True,
        payment_max_skill_multiplier=3,
        pareto_param=4.0,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        # This defines the size of the action space (the max # hours an agent can work).
        self.num_labor_hours = 100  # max 100 hours

        assert isinstance(mask_first_step, bool)
        self.mask_first_step = mask_first_step

        self.is_first_step = True
        self.common_mask_on = {
            agent.idx: np.ones((self.num_labor_hours,)) for agent in self.world.agents
        }
        self.common_mask_off = {
            agent.idx: np.zeros((self.num_labor_hours,)) for agent in self.world.agents
        }

        # Skill distribution
        self.pareto_param = float(pareto_param)
        assert self.pareto_param > 0
        self.payment_max_skill_multiplier = float(payment_max_skill_multiplier)
        pmsm = self.payment_max_skill_multiplier
        num_agents = len(self.world.agents)
        # Generate a batch (1000) of num_agents (sorted/clipped) Pareto samples.
        pareto_samples = np.random.pareto(4, size=(1000, num_agents))
        clipped_skills = np.minimum(pmsm, (pmsm - 1) * pareto_samples + 1)
        sorted_clipped_skills = np.sort(clipped_skills, axis=1)
        # The skill level of the i-th skill-ranked agent is the average of the
        # i-th ranked samples throughout the batch.
        self.skills = sorted_clipped_skills.mean(axis=0)

    def get_additional_state_fields(self, agent_cls_name):
        if agent_cls_name == "BasicMobileAgent":
            return {"skill": 0, "production": 0}
        return {}

    def additional_reset_steps(self):
        self.is_first_step = True
        for agent in self.world.agents:
            agent.state["skill"] = self.skills[agent.idx]

    def get_n_actions(self, agent_cls_name):
        if agent_cls_name == "BasicMobileAgent":
            return self.num_labor_hours
        return None

    def generate_masks(self, completions=0):
        if self.is_first_step:
            self.is_first_step = False
            if self.mask_first_step:
                return self.common_mask_off

        return self.common_mask_on

    def component_step(self):

        for agent in self.world.get_random_order_agents():

            action = agent.get_component_action(self.name)

            if action == 0:  # NO-OP.
                # Agent is not interacting with this component.
                continue

            if 1 <= action <= self.num_labor_hours:  # set reopening phase

                hours_worked = action  # NO-OP is 0 hours.
                agent.state["endogenous"]["Labor"] = hours_worked

                payoff = hours_worked * agent.state["skill"]
                agent.state["production"] += payoff
                agent.inventory["Coin"] += payoff

            else:
                # If action > num_labor_hours, this is an error.
                raise ValueError

    def generate_observations(self):
        obs_dict = dict()
        for agent in self.world.agents:
            obs_dict[str(agent.idx)] = {
                "skill": agent.state["skill"] / self.payment_max_skill_multiplier
            }
        return obs_dict
