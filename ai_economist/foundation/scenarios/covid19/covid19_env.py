# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import json
import os
from datetime import datetime, timedelta

import numpy as np

from ai_economist.foundation.base.base_env import BaseEnvironment, scenario_registry
from ai_economist.foundation.utils import verify_activation_code


@scenario_registry.add
class CovidAndEconomyEnvironment(BaseEnvironment):
    """
    A simulation to model health and economy dynamics amidst the COVID-19 pandemic.
    The environment comprising 51 agents (each agent corresponding to a US state and
    Washington D.C.) and the Federal Government (planner). The state agents decide the
    stringency level of the policy response to the pandemic, while the federal
    government provides subsidies to eligible individuals.

    This simulation makes modeling assumptions. For details, see the technical paper:
    https://arxiv.org/abs/2108.02904

    Args:
        use_real_world_data (bool): Replay what happened in the real world.
            Real-world data comprises SIR (susceptible/infected/recovered),
            unemployment, government policy, and vaccination numbers.
            This setting also sets use_real_world_policies=True.
        use_real_world_policies (bool): Run the environment with real-world policies
            (stringency levels and subsidies). With this setting and
            use_real_world_data=False, SIR and economy dynamics are still
            driven by fitted models.
        path_to_data_and_fitted_params (dirpath): Full path to the directory containing
            the data, fitted parameters and model constants. This defaults to
            "ai_economist/datasets/covid19_datasets/data_and_fitted_params".
            For details on obtaining these parameters, please see the notebook
            "ai-economist-foundation/ai_economist/datasets/covid19_datasets/
            gather_real_world_data_and_fit_parameters.ipynb".
        start_date (string): Date (YYYY-MM-DD) to start the simulation.
        pop_between_age_18_65 (float): Fraction of the population between ages 18-65.
            This is the subset of the population whose employment/unemployment affects
            economic productivity.
            Range: 0 <= pop_between_age_18_65 <= 1.
        infection_too_sick_to_work_rate (float): Fraction of people infected with
            COVID-19. Infected people don't work.
            Range: 0 <= infection_too_sick_to_work_rate <= 1
        risk_free_interest_rate (float): Percentage of interest paid by the federal
            government to borrow money from the federal reserve for COVID-19 relief
            (direct payments). Higher interest rates mean that direct payments
            have a larger cost on the federal government's economic index.
            Range: 0 <= risk_free_interest_rate
        economic_reward_crra_eta (float): CRRA eta parameter for modeling the economic
            reward non-linearity.
            A useful reference: https://en.wikipedia.org/wiki/Isoelastic_utility
            Range: 0 <= economic_reward_crra_eta
        health_priority_scaling_agents (float): A factor indicating how much more the
            states prioritize health (roughly speaking, loss of lives due to
            opening up more) over the economy (roughly speaking, a loss in GDP
            due to shutting down resulting in more unemployment) compared to the
            real-world.
            For example, a value of 1 corresponds to the real-world, while
            a value of 2 means that states cared twice as much about public health
            (preventing deaths), while a value of 0.5 means that states cared twice
            as much about the economy (preventing GDP drops).
            Range: 0 <= health_priority_scaling_agents
        health_priority_scaling_planner (float): same as above,
            but for the federal government.
            Range: 0 <= health_priority_scaling_planner
    """

    def __init__(
        self,
        *base_env_args,
        use_real_world_data=False,
        use_real_world_policies=False,
        path_to_data_and_fitted_params="",
        start_date="2020-03-22",
        pop_between_age_18_65=0.6,
        infection_too_sick_to_work_rate=0.1,
        risk_free_interest_rate=0.03,
        economic_reward_crra_eta=2,
        health_priority_scaling_agents=1,
        health_priority_scaling_planner=1,
        reward_normalization_factor=1,
        **base_env_kwargs,
    ):
        verify_activation_code()

        # Used for datatype checks
        self.np_float_dtype = np.float32
        self.np_int_dtype = np.int32

        # Flag to use real-world data or the fitted models instead
        self.use_real_world_data = use_real_world_data
        # Flag to use real-world policies (actions) or the supplied actions instead
        self.use_real_world_policies = use_real_world_policies

        # If we use real-world data, we also want to use the real-world policies
        if self.use_real_world_data:
            print(
                "Using real-world data to initialize as well as to "
                "step through the env."
            )
            # Note: under this setting, the real_world policies are also used.
            assert self.use_real_world_policies, (
                "Since the env. config. 'use_real_world_data' is True, please also "
                "set 'use_real_world_policies' to True."
            )
        else:
            print(
                "Using the real-world data to only initialize the env, "
                "and using the fitted models to step through the env."
            )

        # Load real-world date
        if path_to_data_and_fitted_params == "":
            current_dir = os.path.dirname(__file__)
            self.path_to_data_and_fitted_params = os.path.join(
                current_dir, "../../../datasets/covid19_datasets/data_and_fitted_params"
            )
        else:
            self.path_to_data_and_fitted_params = path_to_data_and_fitted_params

        print(
            "Loading real-world data from {}".format(
                self.path_to_data_and_fitted_params
            )
        )
        real_world_data_npz = np.load(
            os.path.join(self.path_to_data_and_fitted_params, "real_world_data.npz")
        )
        self._real_world_data = {}
        for key in list(real_world_data_npz):
            self._real_world_data[key] = real_world_data_npz[key]

        # Load fitted parameters
        print(
            "Loading fit parameters from {}".format(self.path_to_data_and_fitted_params)
        )
        self.load_model_constants(self.path_to_data_and_fitted_params)
        self.load_fitted_params(self.path_to_data_and_fitted_params)

        try:
            self.start_date = datetime.strptime(start_date, self.date_format)
        except ValueError:
            print(f"Incorrect data format, should be {self.date_format}")

        # Start date should be beyond the date for which data is available
        assert self.start_date >= self.policy_start_date

        # Compute a start date index based on policy start date
        self.start_date_index = (self.start_date - self.policy_start_date).days
        assert 0 <= self.start_date_index < len(self._real_world_data["policy"])

        # For date logging (This will be overwritten in additional_reset_steps;
        # see below)
        self.current_date = None

        # When using real-world policy, limit the episode length
        # to the length of the available policy.
        if self.use_real_world_policies:
            real_world_policy_length = (
                len(self._real_world_data["policy"]) - self.start_date_index
            )
            print("Using real-world policies, ignoring external action inputs.")
            assert base_env_kwargs["episode_length"] <= real_world_policy_length, (
                f"The real-world policies are only available for "
                f"{real_world_policy_length} timesteps; so the 'episode_length' "
                f"in the environment configuration can only be at most "
                f"{real_world_policy_length}"
            )
        else:
            print("Using external action inputs.")

        # US states and populations
        self.num_us_states = len(self.us_state_population)

        assert (
            base_env_kwargs["n_agents"] == self.num_us_states
        ), "n_agents should be set to the number of US states, i.e., {}.".format(
            self.num_us_states
        )
        # Note: For a faster environment step time, we collate all the individual agents
        # into a single agent index "a" and we flatten the component action masks too.
        assert base_env_kwargs[
            "collate_agent_step_and_reset_data"
        ], "The env. config 'collate_agent_step_and_reset_data' should be set to True."
        super().__init__(*base_env_args, **base_env_kwargs)

        # Add attributes to self.world for use in components
        self.world.us_state_population = self.us_state_population
        self.world.us_population = self.us_population
        self.world.start_date = self.start_date
        self.world.n_stringency_levels = self.num_stringency_levels
        self.world.use_real_world_policies = self.use_real_world_policies
        if self.use_real_world_policies:
            # Agent open/close stringency levels
            self.world.real_world_stringency_policy = self._real_world_data["policy"][
                self.start_date_index :
            ]
            # Planner subsidy levels
            self.world.real_world_subsidy = self._real_world_data["subsidy"][
                self.start_date_index :
            ]

        # Policy --> Unemployment
        #   For accurately modeling the state-wise unemployment, we convolve
        #   the current stringency policy with a family of exponential filters
        #   with separate means (lambdas).
        # This code sets up things we will use in `unemployment_step()`,
        #   which includes a detailed breakdown of how the unemployment model is
        #   implemented.
        self.stringency_level_history = None
        # Each filter captures a temporally extended response to a stringency change.
        self.num_filters = len(self.conv_lambdas)
        self.f_ts = np.tile(
            np.flip(np.arange(self.filter_len), (0,))[None, None],
            (1, self.num_filters, 1),
        ).astype(self.np_float_dtype)
        self.unemp_conv_filters = np.exp(-self.f_ts / self.conv_lambdas[None, :, None])
        # Each state weights these filters differently.
        self.repeated_conv_weights = np.repeat(
            self.grouped_convolutional_filter_weights.reshape(
                self.num_us_states, self.num_filters
            )[:, :, np.newaxis],
            self.filter_len,
            axis=-1,
        )

        # For manually modulating SIR/Unemployment parameters
        self._beta_intercepts_modulation = 1
        self._beta_slopes_modulation = 1
        self._unemployment_modulation = 1

        # Economy-related
        # Interest rate for borrowing money from the federal reserve
        self.risk_free_interest_rate = self.np_float_dtype(risk_free_interest_rate)

        # Compute each worker's daily productivity when at work (to match 2019 GDP)
        # We assume the open/close stringency policy level was always at it's lowest
        # value (i.e., 1) before the pandemic started.
        num_unemployed_at_stringency_level_1 = self.unemployment_step(
            np.ones(self.num_us_states)
        )
        workforce = (
            self.us_population * pop_between_age_18_65
            - np.sum(num_unemployed_at_stringency_level_1)
        ).astype(self.np_int_dtype)
        workers_per_capita = (workforce / self.us_population).astype(
            self.np_float_dtype
        )
        gdp_per_worker = (self.gdp_per_capita / workers_per_capita).astype(
            self.np_float_dtype
        )
        self.num_days_in_an_year = 365
        self.daily_production_per_worker = (
            gdp_per_worker / self.num_days_in_an_year
        ).astype(self.np_float_dtype)

        self.infection_too_sick_to_work_rate = self.np_float_dtype(
            infection_too_sick_to_work_rate
        )
        assert 0 <= self.infection_too_sick_to_work_rate <= 1

        self.pop_between_age_18_65 = self.np_float_dtype(pop_between_age_18_65)
        assert 0 <= self.pop_between_age_18_65 <= 1

        # Compute max possible productivity values (used for agent reward normalization)
        max_productivity_t = self.economy_step(
            self.us_state_population,
            np.zeros((self.num_us_states), dtype=self.np_int_dtype),
            np.zeros((self.num_us_states), dtype=self.np_int_dtype),
            num_unemployed_at_stringency_level_1,
            infection_too_sick_to_work_rate=self.infection_too_sick_to_work_rate,
            population_between_age_18_65=self.pop_between_age_18_65,
        )
        self.maximum_productivity_t = max_productivity_t

        # Economic reward non-linearity
        self.economic_reward_crra_eta = self.np_float_dtype(economic_reward_crra_eta)
        assert 0.0 <= self.economic_reward_crra_eta < 20.0

        # Health indices are normalized by maximum annual GDP
        self.agents_health_norm = self.maximum_productivity_t * self.num_days_in_an_year
        self.planner_health_norm = np.sum(self.agents_health_norm)

        # Economic indices are normalized by maximum annual GDP
        self.agents_economic_norm = (
            self.maximum_productivity_t * self.num_days_in_an_year
        )
        self.planner_economic_norm = np.sum(self.agents_economic_norm)

        def scale_health_over_economic_index(health_priority_scaling, alphas):
            """
            Given starting alpha(s), compute new alphas so that the
            resulting alpha:1-alpha ratio is scaled by health_weightage
            """
            z = alphas / (1 - alphas)  # alphas = z / (1 + z)
            scaled_z = health_priority_scaling * z
            new_alphas = scaled_z / (1 + scaled_z)
            return new_alphas

        # Agents' health and economic index weightages
        # fmt: off
        self.weightage_on_marginal_agent_health_index = \
            scale_health_over_economic_index(
                health_priority_scaling_agents,
                self.inferred_weightage_on_agent_health_index,
            )
        # fmt: on
        assert (
            (self.weightage_on_marginal_agent_health_index >= 0)
            & (self.weightage_on_marginal_agent_health_index <= 1)
        ).all()
        self.weightage_on_marginal_agent_economic_index = (
            1 - self.weightage_on_marginal_agent_health_index
        )

        # Planner's health and economic index weightages
        # fmt: off
        self.weightage_on_marginal_planner_health_index = \
            scale_health_over_economic_index(
                health_priority_scaling_planner,
                self.inferred_weightage_on_planner_health_index,
            )
        # fmt: on
        assert 0 <= self.weightage_on_marginal_planner_health_index <= 1
        self.weightage_on_marginal_planner_economic_index = (
            1 - self.weightage_on_marginal_planner_health_index
        )

        # Normalization factor for the reward (often useful for RL training)
        self.reward_normalization_factor = reward_normalization_factor

    name = "CovidAndEconomySimulation"
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]

    required_entities = []

    def reset_starting_layout(self):
        pass

    def reset_agent_states(self):
        self.world.clear_agent_locs()

    def scenario_step(self):
        """
        Update the state of the USA based on the Covid-19 and Economy dynamics.
        This internally implements three steps
        - sir_step() - updates the susceptible, infected, recovered, deaths
        and vaccination numbers based on the SIR equations
        - unemployment_step() - uses the unemployment model to updates the unemployment
         based on the stringency levels
        - economy_step - computes the current producitivity numbers for the agents
        """
        prev_t = self.world.timestep - 1
        curr_t = self.world.timestep

        self.current_date += timedelta(days=1)

        # SIR
        # ---
        if self.use_real_world_data:
            _S_t = np.maximum(
                self._real_world_data["susceptible"][curr_t + self.start_date_index],
                0,
            )
            _I_t = np.maximum(
                self._real_world_data["infected"][curr_t + self.start_date_index],
                0,
            )
            _R_t = np.maximum(
                self._real_world_data["recovered"][curr_t + self.start_date_index],
                0,
            )
            _V_t = np.maximum(
                self._real_world_data["vaccinated"][curr_t + self.start_date_index],
                0,
            )
            _D_t = np.maximum(
                self._real_world_data["deaths"][curr_t + self.start_date_index],
                0,
            )

        else:  # Use simulation logic
            if curr_t - self.beta_delay < 0:
                if self.start_date_index + curr_t - self.beta_delay < 0:
                    stringency_level_tmk = np.ones(self.num_us_states)
                else:
                    stringency_level_tmk = self._real_world_data["policy"][
                        self.start_date_index + curr_t - self.beta_delay, :
                    ]
            else:
                stringency_level_tmk = self.world.global_state["Stringency Level"][
                    curr_t - self.beta_delay
                ]
            stringency_level_tmk = stringency_level_tmk.astype(self.np_int_dtype)

            _S_tm1 = self.world.global_state["Susceptible"][prev_t]
            _I_tm1 = self.world.global_state["Infected"][prev_t]
            _R_tm1 = self.world.global_state["Recovered"][prev_t]
            _V_tm1 = self.world.global_state["Vaccinated"][prev_t]

            # Vaccination
            # -----------
            num_vaccines_available_t = np.zeros(self.n_agents, dtype=self.np_int_dtype)
            for aidx, agent in enumerate(self.world.agents):
                # "Load" the vaccines in the inventory into this vector.
                num_vaccines_available_t[aidx] = agent.state["Vaccines Available"]
                # Agents always use whatever vaccines they can, so this becomes 0:
                agent.state["Total Vaccinated"] += agent.state["Vaccines Available"]
                agent.state["Vaccines Available"] = 0

            # SIR step
            # --------
            _dS, _dI, _dR, _dV = self.sir_step(
                _S_tm1,
                _I_tm1,
                stringency_level_tmk,
                num_vaccines_available_t,
            )
            _S_t = np.maximum(_S_tm1 + _dS, 0)
            _I_t = np.maximum(_I_tm1 + _dI, 0)
            _R_t = np.maximum(_R_tm1 + _dR, 0)
            _V_t = np.maximum(_V_tm1 + _dV, 0)

            num_recovered_but_not_vaccinated_t = _R_t - _V_t
            _D_t = self.death_rate * num_recovered_but_not_vaccinated_t

        # Update global state
        # -------------------
        self.world.global_state["Susceptible"][curr_t] = _S_t
        self.world.global_state["Infected"][curr_t] = _I_t
        self.world.global_state["Recovered"][curr_t] = _R_t
        self.world.global_state["Deaths"][curr_t] = _D_t
        self.world.global_state["Vaccinated"][curr_t] = _V_t

        # Unemployment
        # ------------
        if self.use_real_world_data:
            num_unemployed_t = self._real_world_data["unemployed"][
                self.start_date_index + curr_t
            ]
        else:
            num_unemployed_t = self.unemployment_step(
                current_stringency_level=self.world.global_state["Stringency Level"][
                    curr_t
                ]
            )

        self.world.global_state["Unemployed"][curr_t] = num_unemployed_t

        # Productivity
        # ------------
        productivity_t = self.economy_step(
            self.us_state_population,
            infected=_I_t,
            deaths=_D_t,
            unemployed=num_unemployed_t,
            infection_too_sick_to_work_rate=self.infection_too_sick_to_work_rate,
            population_between_age_18_65=self.pop_between_age_18_65,
        )

        # Subsidies
        # ---------
        # Add federal government subsidy to productivity
        daily_statewise_subsidy_t = self.world.global_state["Subsidy"][curr_t]
        postsubsidy_productivity_t = productivity_t + daily_statewise_subsidy_t
        self.world.global_state["Postsubsidy Productivity"][
            curr_t
        ] = postsubsidy_productivity_t

        # Update agent state
        # ------------------
        current_date_string = datetime.strftime(
            self.current_date, format=self.date_format
        )
        for agent in self.world.agents:
            agent.state["Total Susceptible"] = _S_t[agent.idx].astype(self.np_int_dtype)
            agent.state["New Infections"] = (
                _I_t[agent.idx] - agent.state["Total Infected"]
            ).astype(self.np_int_dtype)
            agent.state["Total Infected"] = _I_t[agent.idx].astype(self.np_int_dtype)
            agent.state["Total Recovered"] = _R_t[agent.idx].astype(self.np_int_dtype)
            agent.state["New Deaths"] = _D_t[agent.idx] - agent.state[
                "Total Deaths"
            ].astype(self.np_int_dtype)
            agent.state["Total Deaths"] = _D_t[agent.idx].astype(self.np_int_dtype)
            agent.state["Total Vaccinated"] = _V_t[agent.idx].astype(self.np_int_dtype)

            agent.state["Total Unemployed"] = num_unemployed_t[agent.idx].astype(
                self.np_int_dtype
            )
            agent.state["New Subsidy Received"] = daily_statewise_subsidy_t[agent.idx]
            agent.state["Postsubsidy Productivity"] = postsubsidy_productivity_t[
                agent.idx
            ]
            agent.state["Date"] = current_date_string

        # Update planner state
        # --------------------
        self.world.planner.state["Total Susceptible"] = np.sum(_S_t).astype(
            self.np_int_dtype
        )
        self.world.planner.state["New Infections"] = (
            np.sum(_I_t) - self.world.planner.state["Total Infected"]
        ).astype(self.np_int_dtype)
        self.world.planner.state["Total Infected"] = np.sum(_I_t).astype(
            self.np_int_dtype
        )
        self.world.planner.state["Total Recovered"] = np.sum(_R_t).astype(
            self.np_int_dtype
        )
        self.world.planner.state["New Deaths"] = (
            np.sum(_D_t) - self.world.planner.state["Total Deaths"]
        ).astype(self.np_int_dtype)
        self.world.planner.state["Total Deaths"] = np.sum(_D_t).astype(
            self.np_int_dtype
        )
        self.world.planner.state["Total Vaccinated"] = np.sum(_V_t).astype(
            self.np_int_dtype
        )
        self.world.planner.state["Total Unemployed"] = np.sum(num_unemployed_t).astype(
            self.np_int_dtype
        )
        self.world.planner.state["New Subsidy Provided"] = np.sum(
            daily_statewise_subsidy_t
        )
        self.world.planner.state["Postsubsidy Productivity"] = np.sum(
            postsubsidy_productivity_t
        )
        self.world.planner.state["Date"] = current_date_string

    def generate_observations(self):
        """
        - Process agent-specific and planner-specific data into an observation.
        - Observations contain only the relevant features for that actor.
        :return: a dictionary of observations for each agent and planner
        """
        redux_agent_global_state = None
        for feature in [
            "Susceptible",
            "Infected",
            "Recovered",
            "Deaths",
            "Vaccinated",
            "Unemployed",
        ]:
            if redux_agent_global_state is None:
                redux_agent_global_state = self.world.global_state[feature][
                    self.world.timestep
                ]
            else:
                redux_agent_global_state = np.vstack(
                    (
                        redux_agent_global_state,
                        self.world.global_state[feature][self.world.timestep],
                    )
                )
        normalized_redux_agent_state = (
            redux_agent_global_state / self.us_state_population[None]
        )

        # Productivity
        postsubsidy_productivity_t = self.world.global_state[
            "Postsubsidy Productivity"
        ][self.world.timestep]
        normalized_postsubsidy_productivity_t = (
            postsubsidy_productivity_t / self.maximum_productivity_t
        )

        # Let agents know about the policy about to affect SIR infection-rate beta
        t_beta = self.world.timestep - self.beta_delay + 1
        if t_beta < 0:
            lagged_stringency_level = self._real_world_data["policy"][
                self.start_date_index + t_beta
            ]
        else:
            lagged_stringency_level = self.world.global_state["Stringency Level"][
                t_beta
            ]

        normalized_lagged_stringency_level = (
            lagged_stringency_level / self.num_stringency_levels
        )

        # To condition policy on agent id
        agent_index = np.eye(self.n_agents, dtype=self.np_int_dtype)

        # Observation dict - Agents
        # -------------------------
        obs_dict = dict()
        obs_dict["a"] = {
            "agent_index": agent_index,
            "agent_state": normalized_redux_agent_state,
            "agent_postsubsidy_productivity": normalized_postsubsidy_productivity_t,
            "lagged_stringency_level": normalized_lagged_stringency_level,
        }

        # Observation dict - Planner
        # --------------------------
        obs_dict[self.world.planner.idx] = {
            "agent_state": normalized_redux_agent_state,
            "agent_postsubsidy_productivity": normalized_postsubsidy_productivity_t,
            "lagged_stringency_level": normalized_lagged_stringency_level,
        }

        return obs_dict

    def compute_reward(self):
        """
        Compute the social welfare metrics for each agent and the planner.
        :return: a dictionary of rewards for each agent in the simulation
        """
        rew = {"a": 0, "p": 0}

        def crra_nonlinearity(x, eta):
            # Reference: https://en.wikipedia.org/wiki/Isoelastic_utility
            # To be applied to (marginal) economic indices
            annual_x = self.num_days_in_an_year * x
            annual_x_clipped = np.clip(annual_x, 0.1, 3)
            annual_crra = 1 + (annual_x_clipped ** (1 - eta) - 1) / (1 - eta)
            daily_crra = annual_crra / self.num_days_in_an_year
            return daily_crra

        def min_max_normalization(x, min_x, max_x):
            eps = 1e-10
            return (x - min_x) / (max_x - min_x + eps)

        def get_weighted_average(
            health_index_weightage,
            health_index,
            economic_index_weightage,
            economic_index,
        ):
            return (
                health_index_weightage * health_index
                + economic_index_weightage * economic_index
            ) / (health_index_weightage + economic_index_weightage)

        # Changes this last timestep:
        marginal_deaths = (
            self.world.global_state["Deaths"][self.world.timestep]
            - self.world.global_state["Deaths"][self.world.timestep - 1]
        )

        subsidy_t = self.world.global_state["Subsidy"][self.world.timestep]
        postsubsidy_productivity_t = self.world.global_state[
            "Postsubsidy Productivity"
        ][self.world.timestep]

        # Health index -- the cost equivalent (annual GDP) of covid deaths
        # Note: casting deaths to float to prevent overflow issues
        marginal_agent_health_index = (
            -marginal_deaths.astype(self.np_float_dtype)
            * self.value_of_life
            / self.agents_health_norm
        ).astype(self.np_float_dtype)

        # Economic index -- fraction of annual GDP achieved
        # Use a "crra" nonlinearity on the agent economic reward
        marginal_agent_economic_index = crra_nonlinearity(
            postsubsidy_productivity_t / self.agents_economic_norm,
            self.economic_reward_crra_eta,
        ).astype(self.np_float_dtype)

        # Min-max Normalization
        marginal_agent_health_index = min_max_normalization(
            marginal_agent_health_index,
            self.min_marginal_agent_health_index,
            self.max_marginal_agent_health_index,
        ).astype(self.np_float_dtype)
        marginal_agent_economic_index = min_max_normalization(
            marginal_agent_economic_index,
            self.min_marginal_agent_economic_index,
            self.max_marginal_agent_economic_index,
        ).astype(self.np_float_dtype)

        # Agent Rewards
        # -------------
        agent_rewards = get_weighted_average(
            self.weightage_on_marginal_agent_health_index,
            marginal_agent_health_index,
            self.weightage_on_marginal_agent_economic_index,
            marginal_agent_economic_index,
        )
        rew["a"] = agent_rewards / self.reward_normalization_factor

        # Update agent states
        # -------------------
        for agent in self.world.agents:
            agent.state["Health Index"] += marginal_agent_health_index[agent.idx]
            agent.state["Economic Index"] += marginal_agent_economic_index[agent.idx]

        # National level
        # --------------
        # Health index -- the cost equivalent (annual GDP) of covid deaths
        # Note: casting deaths to float to prevent overflow issues
        marginal_planner_health_index = (
            -np.sum(marginal_deaths).astype(self.np_float_dtype)
            * self.value_of_life
            / self.planner_health_norm
        )

        # Economic index -- fraction of annual GDP achieved (minus subsidy cost)
        cost_of_subsidy_t = (1 + self.risk_free_interest_rate) * np.sum(subsidy_t)
        # Use a "crra" nonlinearity on the planner economic reward
        marginal_planner_economic_index = crra_nonlinearity(
            (np.sum(postsubsidy_productivity_t) - cost_of_subsidy_t)
            / self.planner_economic_norm,
            self.economic_reward_crra_eta,
        )

        # Min-max Normalization
        marginal_planner_health_index = min_max_normalization(
            marginal_planner_health_index,
            self.min_marginal_planner_health_index,
            self.max_marginal_planner_health_index,
        )
        marginal_planner_economic_index = min_max_normalization(
            marginal_planner_economic_index,
            self.min_marginal_planner_economic_index,
            self.max_marginal_planner_economic_index,
        )

        # Update planner states
        # -------------------
        self.world.planner.state["Health Index"] += marginal_planner_health_index
        self.world.planner.state["Economic Index"] += marginal_planner_economic_index

        # Planner Reward
        # --------------
        planner_rewards = get_weighted_average(
            self.weightage_on_marginal_planner_health_index,
            marginal_planner_health_index,
            self.weightage_on_marginal_planner_economic_index,
            marginal_planner_economic_index,
        )
        rew[self.world.planner.idx] = planner_rewards / self.reward_normalization_factor

        return rew

    def additional_reset_steps(self):
        assert self.world.timestep == 0

        # Reset current date
        self.current_date = self.start_date

        # SIR numbers at timestep 0
        susceptible_0 = self._real_world_data["susceptible"][self.start_date_index]
        infected_0 = self._real_world_data["infected"][self.start_date_index]
        newly_infected_0 = (
            infected_0
            - self._real_world_data["infected"][max(0, self.start_date_index - 1)]
        )
        recovered_0 = self._real_world_data["recovered"][self.start_date_index]
        deaths_0 = recovered_0 * self.death_rate

        # Unemployment and vaccinated numbers at timestep 0
        unemployed_0 = self._real_world_data["unemployed"][self.start_date_index]
        vaccinated_0 = self._real_world_data["vaccinated"][self.start_date_index]

        # Create a global state dictionary to save episode data
        self.world.global_state = {}
        self.set_global_state("Susceptible", susceptible_0, t=self.world.timestep)
        self.set_global_state("Infected", infected_0, t=self.world.timestep)
        self.set_global_state("Recovered", recovered_0, t=self.world.timestep)
        self.set_global_state("Deaths", deaths_0, t=self.world.timestep)

        self.set_global_state("Unemployed", unemployed_0, t=self.world.timestep)
        self.set_global_state("Vaccinated", vaccinated_0, t=self.world.timestep)

        new_deaths_0 = (
            deaths_0
            - self._real_world_data["recovered"][max(0, self.start_date_index - 1)]
            * self.death_rate
        )

        # Reset stringency level history.
        # Pad with stringency levels of 1 corresponding to states being fully open
        # (as was the case before the pandemic).
        self.stringency_level_history = np.pad(
            self._real_world_data["policy"][: self.start_date_index + 1],
            [(self.filter_len, 0), (0, 0)],
            constant_values=1,
        )[-(self.filter_len + 1) :]

        # Set the stringency level based to the real-world policy
        self.set_global_state(
            "Stringency Level",
            self._real_world_data["policy"][self.start_date_index],
            t=self.world.timestep,
        )

        # All US states start with zero subsidy and zero Postsubsidy Productivity
        self.set_global_state("Subsidy", dtype=self.np_float_dtype)
        self.set_global_state("Postsubsidy Productivity", dtype=self.np_float_dtype)

        # Set initial agent states
        # ------------------------
        current_date_string = datetime.strftime(
            self.current_date, format=self.date_format
        )

        for agent in self.world.agents:
            agent.state["Total Susceptible"] = susceptible_0[agent.idx].astype(
                self.np_int_dtype
            )
            agent.state["New Infections"] = newly_infected_0[agent.idx].astype(
                self.np_int_dtype
            )
            agent.state["Total Infected"] = infected_0[agent.idx].astype(
                self.np_int_dtype
            )
            agent.state["Total Recovered"] = recovered_0[agent.idx].astype(
                self.np_int_dtype
            )
            agent.state["New Deaths"] = new_deaths_0[agent.idx].astype(
                self.np_int_dtype
            )
            agent.state["Total Deaths"] = deaths_0[agent.idx].astype(self.np_int_dtype)
            agent.state["Health Index"] = np.array([0]).astype(self.np_float_dtype)
            agent.state["Economic Index"] = np.array([0]).astype(self.np_float_dtype)
            agent.state["Date"] = current_date_string

        # Planner state fields
        self.world.planner.state["Total Susceptible"] = np.sum(
            [agent.state["Total Susceptible"] for agent in self.world.agents]
        ).astype(self.np_int_dtype)
        self.world.planner.state["New Infections"] = np.sum(
            [agent.state["New Infections"] for agent in self.world.agents]
        ).astype(self.np_int_dtype)
        self.world.planner.state["Total Infected"] = np.sum(
            [agent.state["Total Infected"] for agent in self.world.agents]
        ).astype(self.np_int_dtype)
        self.world.planner.state["Total Recovered"] = np.sum(
            [agent.state["Total Recovered"] for agent in self.world.agents]
        ).astype(self.np_int_dtype)
        self.world.planner.state["New Deaths"] = np.sum(
            [agent.state["New Deaths"] for agent in self.world.agents]
        ).astype(self.np_int_dtype)
        self.world.planner.state["Total Deaths"] = np.sum(
            [agent.state["Total Deaths"] for agent in self.world.agents]
        ).astype(self.np_int_dtype)
        self.world.planner.state["Total Vaccinated"] = np.sum(vaccinated_0).astype(
            self.np_int_dtype
        )
        self.world.planner.state["Health Index"] = np.array([0]).astype(
            self.np_float_dtype
        )
        self.world.planner.state["Economic Index"] = np.array([0]).astype(
            self.np_float_dtype
        )

        self.world.planner.state["Date"] = current_date_string

        # Reset any manually set parameter modulations
        self._beta_intercepts_modulation = 1
        self._beta_slopes_modulation = 1
        self._unemployment_modulation = 1

    def set_global_state(self, key=None, value=None, t=None, dtype=None):
        # Use floats by default for the SIR dynamics
        if dtype is None:
            dtype = self.np_float_dtype
        assert key in [
            "Susceptible",
            "Infected",
            "Recovered",
            "Deaths",
            "Unemployed",
            "Vaccinated",
            "Stringency Level",
            "Subsidy",
            "Postsubsidy Productivity",
        ]
        # If no values are passed, set everything to zeros.
        if key not in self.world.global_state:
            self.world.global_state[key] = np.zeros(
                (self.episode_length + 1, self.num_us_states), dtype=dtype
            )

        if t is not None and value is not None:
            assert isinstance(value, np.ndarray)
            assert value.shape[0] == self.world.global_state[key].shape[1]

            self.world.global_state[key][t] = value
        else:
            pass

    def set_parameter_modulations(
        self, beta_intercept=None, beta_slope=None, unemployment=None
    ):
        """
        Apply parameter modulation, which will be in effect until the next env reset.

        Each modulation term scales the associated set of model parameters by the
        input value. This method is useful for performing a sensitivity analysis.

        In effect, the transmission rate (beta) will be calculated as:
            beta = (m_s * beta_slope)*lagged_stringency + (m_i * beta_intercept)

        The unemployment rate (u) will be calculated as:
            u = SOFTPLUS( m_u * SUM(u_filter_weight * u_filter_response) ) + u_0

        Args:
             beta_intercept: (float, >= 0) Modulation applied to the intercept term
             of the beta model, m_i in above equations
             beta_slope: (float, >= 0) Modulation applied to the slope term of the
             beta model, m_s in above equations
             unemployment: (float, >= 0) Modulation applied to the weighted sum of
             unemployment filter responses, m_u in above equations.

        Example:
            # Reset the environment
            env.reset()

            # Increase the slope of the beta response by 15%
            env.set_parameter_modulations(beta_slope=1.15)

            # Run the environment (this example skips over action selection for brevity)
            for t in range(env.episode_length):
                env.step(actions[t])
        """
        if beta_intercept is not None:
            beta_intercept = float(beta_intercept)
            assert beta_intercept >= 0
            self._beta_intercepts_modulation = beta_intercept

        if beta_slope is not None:
            beta_slope = float(beta_slope)
            assert beta_slope >= 0
            self._beta_slopes_modulation = beta_slope

        if unemployment is not None:
            unemployment = float(unemployment)
            assert unemployment >= 0
            self._unemployment_modulation = unemployment

    def unemployment_step(self, current_stringency_level):
        """
        Computes unemployment given the current stringency level and past levels.

        Unemployment is computed as follows:
        1) For each of self.num_filters, an exponentially decaying filter is
        convolved with the history of stringency changes. Responses move forward in
        time, so a stringency change at time t-1 impacts the response at time t.
        2) The filter responses at time t (the current timestep) are summed together
        using state-specific weights.
        3) The weighted sum is passed through a SOFTPLUS function to capture excess
        unemployment due to stringency policy.
        4) The excess unemployment is added to a state-specific baseline unemployment
        level to get the total unemployment.

        Note: Internally, unemployment is computed somewhat differently for speed.
            In particular, no convolution is used. Instead the "filter response" at
            time t is just a temporally discounted sum of past stringency changes,
            with the discounting given by the filter decay rate.
        """

        def softplus(x, beta=1, threshold=20):
            """
            Numpy implementation of softplus. For reference, see
            https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
            """
            return 1 / beta * np.log(1 + np.exp(beta * x)) * (
                beta * x <= threshold
            ) + x * (beta * x > threshold)

        if (
            self.world.timestep == 0
        ):  # computing unemployment at closure policy "all ones"
            delta_stringency_level = np.zeros((self.filter_len, self.num_us_states))
        else:
            self.stringency_level_history = np.concatenate(
                (
                    self.stringency_level_history[1:],
                    current_stringency_level.reshape(1, -1),
                )
            )
            delta_stringency_level = (
                self.stringency_level_history[1:] - self.stringency_level_history[:-1]
            )

        # Rather than modulating the unemployment params,
        # modulate the deltas (same effect)
        delta_stringency_level = delta_stringency_level * self._unemployment_modulation

        # Expand the [time, state] delta history to have a dimension for filter channel
        x_data = delta_stringency_level[None].transpose(2, 0, 1)

        # Apply the state-specific filter weights to each channel
        weighted_x_data = x_data * self.repeated_conv_weights

        # Compute the discounted sum of the weighted deltas, with each channel using
        # a discounting rate reflecting the time constant of the filter channel. Also
        # sum over channels and use a softplus to get excess unemployment.
        excess_unemployment = softplus(
            np.sum(weighted_x_data * self.unemp_conv_filters, axis=(1, 2)), beta=1
        )

        # Add excess unemployment to baseline unemployment
        unemployment_rate = excess_unemployment + self.unemployment_bias

        # Convert the rate (which is a percent) to raw numbers for output
        num_unemployed_t = unemployment_rate * self.us_state_population / 100
        return num_unemployed_t

    # --- Scenario-specific ---
    def economy_step(
        self,
        population,
        infected,
        deaths,
        unemployed,
        infection_too_sick_to_work_rate=0.05,
        population_between_age_18_65=0.67,
    ):
        """
        Computes how much production occurs.

        Assumptions:

        - People that cannot work: "infected + aware" and "unemployed" and "deaths".
        - No life/death cycles.

        See __init__() for pre-computation of each worker's daily productivity.
        """

        incapacitated = (infection_too_sick_to_work_rate * infected) + deaths
        cant_work = (incapacitated * population_between_age_18_65) + unemployed

        num_workers = population * population_between_age_18_65

        num_people_that_can_work = np.maximum(0, num_workers - cant_work)

        productivity = (
            num_people_that_can_work * self.daily_production_per_worker
        ).astype(self.np_float_dtype)

        return productivity

    def sir_step(
        self,
        S_tm1,
        I_tm1,
        stringency_level_tmk,
        num_vaccines_available_t,
    ):
        """
        Simulates SIR infection model in the US.
        """
        intercepts = self.beta_intercepts * self._beta_intercepts_modulation
        slopes = self.beta_slopes * self._beta_slopes_modulation
        beta_i = (intercepts + slopes * stringency_level_tmk).astype(
            self.np_float_dtype
        )

        small_number = 1e-10  # used to prevent indeterminate cases
        susceptible_fraction_vaccinated = np.minimum(
            np.ones((self.num_us_states), dtype=self.np_int_dtype),
            num_vaccines_available_t / (S_tm1 + small_number),
        ).astype(self.np_float_dtype)
        vaccinated_t = np.minimum(num_vaccines_available_t, S_tm1)

        # Record R0
        R0 = beta_i / self.gamma
        for agent in self.world.agents:
            agent.state["R0"] = R0[agent.idx]

        # S -> I; dS
        neighborhood_SI_over_N = (S_tm1 / self.us_state_population) * I_tm1
        dS_t = (
            -beta_i * neighborhood_SI_over_N * (1 - susceptible_fraction_vaccinated)
            - vaccinated_t
        ).astype(self.np_float_dtype)

        # I -> R; dR
        dR_t = (self.gamma * I_tm1 + vaccinated_t).astype(self.np_float_dtype)

        # dI from d(S + I + R) = 0
        # ------------------------
        dI_t = -dS_t - dR_t

        dV_t = vaccinated_t.astype(self.np_float_dtype)

        return dS_t, dI_t, dR_t, dV_t

    def load_model_constants(self, path_to_model_constants):
        filename = "model_constants.json"
        assert filename in os.listdir(path_to_model_constants), (
            "Unable to locate '{}' in '{}'.\nPlease run the "
            "'gather_real_world_data.ipynb' notebook first".format(
                filename, path_to_model_constants
            )
        )
        with open(os.path.join(path_to_model_constants, filename), "r") as fp:
            model_constants_dict = json.load(fp)
        fp.close()

        self.date_format = model_constants_dict["DATE_FORMAT"]
        self.us_state_idx_to_state_name = model_constants_dict[
            "US_STATE_IDX_TO_STATE_NAME"
        ]
        self.us_state_population = self.np_int_dtype(
            model_constants_dict["US_STATE_POPULATION"]
        )
        self.us_population = self.np_int_dtype(model_constants_dict["US_POPULATION"])
        self.num_stringency_levels = model_constants_dict["NUM_STRINGENCY_LEVELS"]
        self.death_rate = self.np_float_dtype(model_constants_dict["SIR_MORTALITY"])
        self.gamma = self.np_float_dtype(model_constants_dict["SIR_GAMMA"])
        self.gdp_per_capita = self.np_float_dtype(
            model_constants_dict["GDP_PER_CAPITA"]
        )

    def load_fitted_params(self, path_to_fitted_params):
        filename = "fitted_params.json"
        assert filename in os.listdir(path_to_fitted_params), (
            "Unable to locate '{}' in '{}'.\nIf you ran the "
            "'gather_real_world_data.ipynb' notebook to download the latest "
            "real-world data, please also run the "
            "'fit_parameters.ipynb' notebook.".format(filename, path_to_fitted_params)
        )
        with open(os.path.join(path_to_fitted_params, filename), "r") as fp:
            fitted_params_dict = json.load(fp)
        fp.close()
        self.policy_start_date = datetime.strptime(
            fitted_params_dict["POLICY_START_DATE"], self.date_format
        )
        self.value_of_life = self.np_int_dtype(fitted_params_dict["VALUE_OF_LIFE"])
        self.beta_delay = self.np_int_dtype(fitted_params_dict["BETA_DELAY"])
        self.beta_slopes = np.array(
            fitted_params_dict["BETA_SLOPES"], dtype=self.np_float_dtype
        )
        self.beta_intercepts = np.array(
            fitted_params_dict["BETA_INTERCEPTS"], dtype=self.np_float_dtype
        )
        self.min_marginal_agent_health_index = np.array(
            fitted_params_dict["MIN_MARGINAL_AGENT_HEALTH_INDEX"],
            dtype=self.np_float_dtype,
        )
        self.max_marginal_agent_health_index = np.array(
            fitted_params_dict["MAX_MARGINAL_AGENT_HEALTH_INDEX"],
            dtype=self.np_float_dtype,
        )
        self.min_marginal_agent_economic_index = np.array(
            fitted_params_dict["MIN_MARGINAL_AGENT_ECONOMIC_INDEX"],
            dtype=self.np_float_dtype,
        )
        self.max_marginal_agent_economic_index = np.array(
            fitted_params_dict["MAX_MARGINAL_AGENT_ECONOMIC_INDEX"],
            dtype=self.np_float_dtype,
        )
        self.min_marginal_planner_health_index = self.np_float_dtype(
            fitted_params_dict["MIN_MARGINAL_PLANNER_HEALTH_INDEX"]
        )
        self.max_marginal_planner_health_index = self.np_float_dtype(
            fitted_params_dict["MAX_MARGINAL_PLANNER_HEALTH_INDEX"]
        )
        self.min_marginal_planner_economic_index = self.np_float_dtype(
            fitted_params_dict["MIN_MARGINAL_PLANNER_ECONOMIC_INDEX"]
        )
        self.max_marginal_planner_economic_index = self.np_float_dtype(
            fitted_params_dict["MAX_MARGINAL_PLANNER_ECONOMIC_INDEX"]
        )
        self.inferred_weightage_on_agent_health_index = np.array(
            fitted_params_dict["INFERRED_WEIGHTAGE_ON_AGENT_HEALTH_INDEX"],
            dtype=self.np_float_dtype,
        )
        self.inferred_weightage_on_planner_health_index = self.np_float_dtype(
            fitted_params_dict["INFERRED_WEIGHTAGE_ON_PLANNER_HEALTH_INDEX"]
        )
        self.filter_len = self.np_int_dtype(fitted_params_dict["FILTER_LEN"])
        self.conv_lambdas = np.array(
            fitted_params_dict["CONV_LAMBDAS"], dtype=self.np_float_dtype
        )
        self.unemployment_bias = np.array(
            fitted_params_dict["UNEMPLOYMENT_BIAS"], dtype=self.np_float_dtype
        )
        self.grouped_convolutional_filter_weights = np.array(
            fitted_params_dict["GROUPED_CONVOLUTIONAL_FILTER_WEIGHTS"],
            dtype=self.np_float_dtype,
        )

    def scenario_metrics(self):
        # End of episode metrics
        # ----------------------
        metrics_dict = {}

        # State-level metrics
        for agent in self.world.agents:
            state_name = self.us_state_idx_to_state_name[str(agent.idx)]

            for field in [
                "infected",
                "recovered",
                "deaths",
            ]:
                metric_key = "{}/{} (millions)".format(state_name, field)
                metrics_dict[metric_key] = (
                    agent.state["Total " + field.capitalize()] / 1e6
                )

            metrics_dict["{}/mean_unemployment_rate (%)".format(state_name)] = (
                np.mean(
                    self.world.global_state["Unemployed"][1:, agent.idx],
                    axis=0,
                )
                / self.us_state_population[agent.idx]
                * 100
            )

            metrics_dict[
                "{}/mean_open_close_stringency_level".format(state_name)
            ] = np.mean(
                self.world.global_state["Stringency Level"][1:, agent.idx],
                axis=0,
            )

            metrics_dict["{}/total_productivity (billion $)".format(state_name)] = (
                np.sum(
                    self.world.global_state["Postsubsidy Productivity"][1:, agent.idx],
                )
                / 1e9
            )

            metrics_dict[
                "{}/health_index_at_end_of_episode".format(state_name)
            ] = agent.state["Health Index"]
            metrics_dict[
                "{}/economic_index_at_end_of_episode".format(state_name)
            ] = agent.state["Economic Index"]

        # USA-level metrics
        metrics_dict["usa/vaccinated (% of population)"] = (
            np.sum(
                self.world.global_state["Vaccinated"][self.world.timestep],
                axis=0,
            )
            / self.us_population
            * 100
        )
        metrics_dict["usa/deaths (thousands)"] = (
            np.sum(
                self.world.global_state["Deaths"][self.world.timestep],
                axis=0,
            )
            / 1e3
        )

        metrics_dict["usa/mean_unemployment_rate (%)"] = (
            np.mean(
                np.sum(
                    self.world.global_state["Unemployed"][1:],
                    axis=1,
                )
                / self.us_population,
                axis=0,
            )
            * 100
        )
        metrics_dict["usa/total_amount_subsidized (trillion $)"] = (
            np.sum(
                self.world.global_state["Subsidy"][1:],
                axis=(0, 1),
            )
            / 1e12
        )
        metrics_dict["usa/total_productivity (trillion $)"] = (
            np.sum(
                self.world.global_state["Postsubsidy Productivity"][1:],
                axis=(0, 1),
            )
            / 1e12
        )

        metrics_dict["usa/health_index_at_end_of_episode"] = self.world.planner.state[
            "Health Index"
        ]
        metrics_dict["usa/economic_index_at_end_of_episode"] = self.world.planner.state[
            "Economic Index"
        ]

        return metrics_dict
