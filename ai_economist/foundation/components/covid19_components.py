# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from datetime import datetime

import numpy as np

from ai_economist.foundation.base.base_component import (
    BaseComponent,
    component_registry,
)


@component_registry.add
class ControlUSStateOpenCloseStatus(BaseComponent):
    """
    Sets the open/close stringency levels for states.
    Args:
        n_stringency_levels (int): number of stringency levels the states can chose
            from. (Must match the number in the model constants dictionary referenced by
            the parent scenario.)
        action_cooldown_period (int): action cooldown period in days.
            Once a stringency level is set, the state(s) cannot switch to another level
            for a certain number of days (referred to as the "action_cooldown_period")
    """

    name = "ControlUSStateOpenCloseStatus"
    required_entities = []
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
        self,
        *base_component_args,
        n_stringency_levels=10,
        action_cooldown_period=28,
        **base_component_kwargs,
    ):

        self.action_cooldown_period = action_cooldown_period
        super().__init__(*base_component_args, **base_component_kwargs)
        self.np_int_dtype = np.int32

        self.n_stringency_levels = int(n_stringency_levels)
        assert self.n_stringency_levels >= 2
        self._checked_n_stringency_levels = False

        self.masks = dict()
        self.default_agent_action_mask = [1 for _ in range(self.n_stringency_levels)]
        self.no_op_agent_action_mask = [0 for _ in range(self.n_stringency_levels)]
        self.masks["a"] = np.repeat(
            np.array(self.no_op_agent_action_mask)[:, np.newaxis],
            self.n_agents,
            axis=-1,
        )

        # (This will be overwritten during reset; see below)
        self.action_in_cooldown_until = None

    def get_additional_state_fields(self, agent_cls_name):
        return {}

    def additional_reset_steps(self):
        # Store the times when the next set of actions can be taken.
        self.action_in_cooldown_until = np.array(
            [self.world.timestep for _ in range(self.n_agents)]
        )

    def get_n_actions(self, agent_cls_name):
        if agent_cls_name == "BasicMobileAgent":
            return self.n_stringency_levels
        return None

    def generate_masks(self, completions=0):
        for agent in self.world.agents:
            if self.world.use_real_world_policies:
                self.masks["a"][:, agent.idx] = self.default_agent_action_mask
            else:
                if self.world.timestep < self.action_in_cooldown_until[agent.idx]:
                    # Keep masking the actions
                    self.masks["a"][:, agent.idx] = self.no_op_agent_action_mask
                else:  # self.world.timestep == self.action_in_cooldown_until[agent.idx]
                    # Cooldown period has ended; unmask the "subsequent" action
                    self.masks["a"][:, agent.idx] = self.default_agent_action_mask
        return self.masks

    def component_step(self):
        if not self._checked_n_stringency_levels:
            if self.n_stringency_levels != self.world.n_stringency_levels:
                raise ValueError(
                    "The environment was not configured correctly. For the given "
                    "model fit, you need to set the number of stringency levels to "
                    "be {}".format(self.world.n_stringency_levels)
                )
            self._checked_n_stringency_levels = True

        for agent in self.world.agents:
            if self.world.use_real_world_policies:
                # Use the action taken in the previous timestep
                action = self.world.real_world_stringency_policy[
                    self.world.timestep - 1, agent.idx
                ]
            else:
                action = agent.get_component_action(self.name)
            assert 0 <= action <= self.n_stringency_levels

            # We only update the stringency level if the action is not a NO-OP.
            self.world.global_state["Stringency Level"][
                self.world.timestep, agent.idx
            ] = (
                self.world.global_state["Stringency Level"][
                    self.world.timestep - 1, agent.idx
                ]
                * (action == 0)
                + action
            )

            agent.state[
                "Current Open Close Stringency Level"
            ] = self.world.global_state["Stringency Level"][
                self.world.timestep, agent.idx
            ]

            # Check if the action cooldown period has ended, and set the next time until
            # action cooldown. If current action is a no-op (i.e., no new action was
            # taken), the agent can take an action in the very next step, otherwise it
            # needs to wait for self.action_cooldown_period steps.
            # When in the action cooldown period, whatever actions the agents take are
            # masked out, so it's always a NO-OP (see generate_masks() above)
            # The logic below influences the action masks.
            if self.world.timestep == self.action_in_cooldown_until[agent.idx] + 1:
                if action == 0:  # NO-OP
                    self.action_in_cooldown_until[agent.idx] += 1
                else:
                    self.action_in_cooldown_until[
                        agent.idx
                    ] += self.action_cooldown_period

    def generate_observations(self):

        # Normalized observations
        obs_dict = dict()
        agent_policy_indicators = self.world.global_state["Stringency Level"][
            self.world.timestep
        ]
        obs_dict["a"] = {
            "agent_policy_indicators": agent_policy_indicators
            / self.n_stringency_levels
        }
        obs_dict[self.world.planner.idx] = {
            "agent_policy_indicators": agent_policy_indicators
            / self.n_stringency_levels
        }

        return obs_dict


@component_registry.add
class FederalGovernmentSubsidy(BaseComponent):
    """
    Args:
        subsidy_interval (int): The number of days over which the total subsidy amount
            is evenly rolled out.
            Note: shortening the subsidy interval increases the total amount of money
            that the planner could possibly spend. For instance, if the subsidy
            interval is 30, the planner can create a subsidy every 30 days.
        num_subsidy_levels (int): The number of subsidy levels.
            Note: with max_annual_subsidy_per_person=10000, one round of subsidies at
            the maximum subsidy level equals an expenditure of roughly $3.3 trillion
            (given the US population of 330 million).
            If the planner chooses the maximum subsidy amount, the $3.3 trillion
            is rolled out gradually over the subsidy interval.
        max_annual_subsidy_per_person (float): The maximum annual subsidy that may be
            allocated per person.
    """

    name = "FederalGovernmentSubsidy"
    required_entities = []
    agent_subclasses = ["BasicPlanner"]

    def __init__(
        self,
        *base_component_args,
        subsidy_interval=90,
        num_subsidy_levels=20,
        max_annual_subsidy_per_person=20000,
        **base_component_kwargs,
    ):
        self.subsidy_interval = int(subsidy_interval)
        assert self.subsidy_interval >= 1

        self.num_subsidy_levels = int(num_subsidy_levels)
        assert self.num_subsidy_levels >= 1

        self.max_annual_subsidy_per_person = float(max_annual_subsidy_per_person)
        assert self.max_annual_subsidy_per_person >= 0

        self.np_int_dtype = np.int32

        # (This will be overwritten during component_step; see below)
        self._subsidy_amount_per_level = None
        self._subsidy_level_array = None

        super().__init__(*base_component_args, **base_component_kwargs)

        self.default_planner_action_mask = [1 for _ in range(self.num_subsidy_levels)]
        self.no_op_planner_action_mask = [0 for _ in range(self.num_subsidy_levels)]

        # (This will be overwritten during reset; see below)
        self.max_daily_subsidy_per_state = np.array(
            self.n_agents, dtype=self.np_int_dtype
        )

    def get_additional_state_fields(self, agent_cls_name):
        if agent_cls_name == "BasicPlanner":
            return {"Total Subsidy": 0, "Current Subsidy Level": 0}
        return {}

    def additional_reset_steps(self):
        # Pre-compute maximum state-specific subsidy levels
        self.max_daily_subsidy_per_state = (
            self.world.us_state_population * self.max_annual_subsidy_per_person / 365
        )

    def get_n_actions(self, agent_cls_name):
        if agent_cls_name == "BasicPlanner":
            # Number of non-zero subsidy levels
            # (the action 0 pertains to the no-subsidy case)
            return self.num_subsidy_levels
        return None

    def generate_masks(self, completions=0):
        masks = {}
        if self.world.use_real_world_policies:
            masks[self.world.planner.idx] = self.default_planner_action_mask
        else:
            if self.world.timestep % self.subsidy_interval == 0:
                masks[self.world.planner.idx] = self.default_planner_action_mask
            else:
                masks[self.world.planner.idx] = self.no_op_planner_action_mask
        return masks

    def component_step(self):
        if self.world.use_real_world_policies:
            if self._subsidy_amount_per_level is None:
                self._subsidy_amount_per_level = (
                    self.world.us_population
                    * self.max_annual_subsidy_per_person
                    / self.num_subsidy_levels
                    * self.subsidy_interval
                    / 365
                )
                self._subsidy_level_array = np.zeros((self._episode_length + 1))
            # Use the action taken in the previous timestep
            current_subsidy_amount = self.world.real_world_subsidy[
                self.world.timestep - 1
            ]
            if current_subsidy_amount > 0:
                _subsidy_level = np.round(
                    (current_subsidy_amount / self._subsidy_amount_per_level)
                )
                for t_idx in range(
                    self.world.timestep - 1,
                    min(
                        len(self._subsidy_level_array),
                        self.world.timestep - 1 + self.subsidy_interval,
                    ),
                ):
                    self._subsidy_level_array[t_idx] += _subsidy_level
            subsidy_level = self._subsidy_level_array[self.world.timestep - 1]
        else:
            # Update the subsidy level only every self.subsidy_interval, since the
            # other actions are masked out.
            if (self.world.timestep - 1) % self.subsidy_interval == 0:
                subsidy_level = self.world.planner.get_component_action(self.name)
            else:
                subsidy_level = self.world.planner.state["Current Subsidy Level"]

        assert 0 <= subsidy_level <= self.num_subsidy_levels
        self.world.planner.state["Current Subsidy Level"] = np.array(
            subsidy_level
        ).astype(self.np_int_dtype)

        # Update subsidy level
        subsidy_level_frac = subsidy_level / self.num_subsidy_levels
        daily_statewise_subsidy = subsidy_level_frac * self.max_daily_subsidy_per_state

        self.world.global_state["Subsidy"][
            self.world.timestep
        ] = daily_statewise_subsidy
        self.world.planner.state["Total Subsidy"] += np.sum(daily_statewise_subsidy)

    def generate_observations(self):
        # Allow the agents/planner to know when the next subsidy might come.
        # Obs should = 0 when the next timestep could include a subsidy
        t_since_last_subsidy = self.world.timestep % self.subsidy_interval
        # (this is normalized to 0<-->1)
        t_until_next_subsidy = self.subsidy_interval - t_since_last_subsidy
        t_vec = t_until_next_subsidy * np.ones(self.n_agents)

        current_subsidy_level = self.world.planner.state["Current Subsidy Level"]
        sl_vec = current_subsidy_level * np.ones(self.n_agents)

        # Normalized observations
        obs_dict = dict()
        obs_dict["a"] = {
            "t_until_next_subsidy": t_vec / self.subsidy_interval,
            "current_subsidy_level": sl_vec / self.num_subsidy_levels,
        }
        obs_dict[self.world.planner.idx] = {
            "t_until_next_subsidy": t_until_next_subsidy / self.subsidy_interval,
            "current_subsidy_level": current_subsidy_level / self.num_subsidy_levels,
        }

        return obs_dict


@component_registry.add
class VaccinationCampaign(BaseComponent):
    """
    Implements a (passive) component for delivering vaccines to agents once a certain
    amount of time has elapsed.

    Args:
        daily_vaccines_per_million_people (int): The number of vaccines available per
            million people everyday.
        delivery_interval (int): The number of days between vaccine deliveries.
        vaccine_delivery_start_date (string): The date (YYYY-MM-DD) when the
            vaccination begins.
    """

    name = "VaccinationCampaign"
    required_entities = []
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
        self,
        *base_component_args,
        daily_vaccines_per_million_people=4500,
        delivery_interval=1,
        vaccine_delivery_start_date="2020-12-22",
        observe_rate=False,
        **base_component_kwargs,
    ):
        self.daily_vaccines_per_million_people = int(daily_vaccines_per_million_people)
        assert 0 <= self.daily_vaccines_per_million_people <= 1e6

        self.delivery_interval = int(delivery_interval)
        assert 1 <= self.delivery_interval <= 5000

        try:
            self.vaccine_delivery_start_date = datetime.strptime(
                vaccine_delivery_start_date, "%Y-%m-%d"
            )
        except ValueError:
            print("Incorrect data format, should be YYYY-MM-DD")

        # (This will  be overwritten during component_step (see below))
        self._time_when_vaccine_delivery_begins = None

        self.np_int_dtype = np.int32

        self.observe_rate = bool(observe_rate)

        super().__init__(*base_component_args, **base_component_kwargs)

        # (This will be overwritten during reset; see below)
        self._num_vaccines_per_delivery = None
        # Convenience for obs (see usage below):
        self._t_first_delivery = None

    @property
    def num_vaccines_per_delivery(self):
        if self._num_vaccines_per_delivery is None:
            # Pre-compute dispersal numbers
            millions_of_residents = self.world.us_state_population / 1e6
            daily_vaccines = (
                millions_of_residents * self.daily_vaccines_per_million_people
            )
            num_vaccines_per_delivery = np.floor(
                self.delivery_interval * daily_vaccines
            )
            self._num_vaccines_per_delivery = np.array(
                num_vaccines_per_delivery, dtype=self.np_int_dtype
            )
        return self._num_vaccines_per_delivery

    @property
    def time_when_vaccine_delivery_begins(self):
        if self._time_when_vaccine_delivery_begins is None:
            self._time_when_vaccine_delivery_begins = (
                self.vaccine_delivery_start_date - self.world.start_date
            ).days
        return self._time_when_vaccine_delivery_begins

    def get_additional_state_fields(self, agent_cls_name):
        if agent_cls_name == "BasicMobileAgent":
            return {"Total Vaccinated": 0, "Vaccines Available": 0}
        return {}

    def additional_reset_steps(self):
        pass

    def get_n_actions(self, agent_cls_name):
        return  # Passive component

    def generate_masks(self, completions=0):
        return {}  # Passive component

    def component_step(self):
        # Do nothing if vaccines are not available yet
        if self.world.timestep < self.time_when_vaccine_delivery_begins:
            return

        # Do nothing if this is not the start of a delivery interval.
        # Vaccines are delivered at the start of each interval.
        if (self.world.timestep % self.delivery_interval) != 0:
            return

        # Deliver vaccines to each state
        for aidx, vaccines in enumerate(self.num_vaccines_per_delivery):
            self.world.agents[aidx].state["Vaccines Available"] += vaccines

    def generate_observations(self):
        # Allow the agents/planner to know when the next vaccines might come.
        # Obs should = 0 when the next timestep will deliver vaccines
        # (this is normalized to 0<-->1)

        if self._t_first_delivery is None:
            self._t_first_delivery = int(self.time_when_vaccine_delivery_begins)
            while (self._t_first_delivery % self.delivery_interval) != 0:
                self._t_first_delivery += 1

        next_t = self.world.timestep + 1
        if next_t <= self._t_first_delivery:
            t_until_next_vac = np.minimum(
                1, (self._t_first_delivery - next_t) / self.delivery_interval
            )
            next_vax_rate = 0.0
        else:
            t_since_last_vac = next_t % self.delivery_interval
            t_until_next_vac = self.delivery_interval - t_since_last_vac
            next_vax_rate = self.daily_vaccines_per_million_people / 1e6
        t_vec = t_until_next_vac * np.ones(self.n_agents)
        r_vec = next_vax_rate * np.ones(self.n_agents)

        # Normalized observations
        obs_dict = dict()
        obs_dict["a"] = {
            "t_until_next_vaccines": t_vec / self.delivery_interval,
        }
        obs_dict[self.world.planner.idx] = {
            "t_until_next_vaccines": t_until_next_vac / self.delivery_interval,
        }

        if self.observe_rate:
            obs_dict["a"]["next_vaccination_rate"] = r_vec
            obs_dict["p"]["next_vaccination_rate"] = float(next_vax_rate)

        return obs_dict
