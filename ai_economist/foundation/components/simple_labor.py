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
    Implements how Agents' labor yields income.
    """

    name = "SimpleLabor"
    required_entities = ["Coin"]
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
        self,
        *base_component_args,
        mask_first_step=True,
        payment_max_skill_multiplier=3,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)
        self.num_labor_hours = 100  # max 100 hours / week to work

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
        self.payment_max_skill_multiplier = float(payment_max_skill_multiplier)
        pmsm = self.payment_max_skill_multiplier
        num_agents = len(self.world.agents)
        ranked_skills = np.array(
            [
                np.sort(
                    np.minimum(
                        pmsm, (pmsm - 1) * np.random.pareto(4, size=num_agents) + 1
                    )
                )
                for _ in range(1000)
            ]
        )
        self.skills = ranked_skills.mean(axis=0)

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
                # We only declared 1 action for this agent type,
                # so action > 1 is an error.
                raise ValueError

    def generate_observations(self):
        obs_dict = dict()
        for agent in self.world.agents:
            obs_dict[str(agent.idx)] = {
                "skill": agent.state["skill"] / self.payment_max_skill_multiplier
            }
        return obs_dict
