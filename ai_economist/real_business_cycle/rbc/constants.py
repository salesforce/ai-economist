# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import torch

_NP_DTYPE = np.float32


def all_agents_export_experiment_template(
    NUMFIRMS, NUMCONSUMERS, NUMGOVERNMENTS, episodes_const=30000
):
    consumption_choices = [
        np.array([0.0 + 1.0 * c for c in range(11)], dtype=_NP_DTYPE)
    ]
    work_choices = [
        np.array([0.0 + 20 * 13 * h for h in range(5)], dtype=_NP_DTYPE)
    ]  # specify dtype -- be consistent?

    consumption_choices = np.array(
        list(itertools.product(*consumption_choices)), dtype=_NP_DTYPE
    )
    work_choices = np.array(list(itertools.product(*work_choices)), dtype=_NP_DTYPE)

    price_choices = np.array([0.0 + 500.0 * c for c in range(6)], dtype=_NP_DTYPE)
    wage_choices = np.array([0.0, 11.0, 22.0, 33.0, 44.0], dtype=_NP_DTYPE)
    capital_choices = np.array([0.1], dtype=_NP_DTYPE)
    price_and_wage = np.array(
        list(itertools.product(price_choices, wage_choices, capital_choices)),
        dtype=_NP_DTYPE,
    )

    # government action discretization
    income_taxation_choices = np.array(
        [0.0 + 0.2 * c for c in range(6)], dtype=_NP_DTYPE
    )
    corporate_taxation_choices = np.array(
        [0.0 + 0.2 * c for c in range(6)], dtype=_NP_DTYPE
    )
    tax_choices = np.array(
        list(itertools.product(income_taxation_choices, corporate_taxation_choices)),
        dtype=_NP_DTYPE,
    )
    global_state_dim = (
        NUMFIRMS  # prices
        + NUMFIRMS  # wages
        + NUMFIRMS  # stocks
        + NUMFIRMS  # was good overdemanded
        + 2 * NUMGOVERNMENTS  # tax rates
        + 1
    )  # time

    global_state_digit_dims = list(
        range(2 * NUMFIRMS, 3 * NUMFIRMS)
    )  # stocks are the only global state var that can get huge
    consumer_state_dim = (
        global_state_dim + 1 + 1
    )  # budget  # theta, the disutility of work

    firm_state_dim = (
        global_state_dim
        + 1  # budget
        + 1  # capital
        + 1  # production alpha
        + NUMFIRMS  # onehot specifying which firm
    )

    episodes_to_anneal_firm = 100000
    episodes_to_anneal_government = 100000
    government_phase1_start = 100000
    government_state_dim = global_state_dim
    DEFAULT_CFG_DICT = {
        # actions_array key will be added below
        "agents": {
            "num_consumers": NUMCONSUMERS,
            "num_firms": NUMFIRMS,
            "num_governments": NUMGOVERNMENTS,
            "global_state_dim": global_state_dim,
            "consumer_state_dim": consumer_state_dim,
            # action vectors are how much consume from each firm,
            # how much to work, and which firm to choose
            "consumer_action_dim": NUMFIRMS + 1 + 1,
            "consumer_num_consume_actions": consumption_choices.shape[0],
            "consumer_num_work_actions": work_choices.shape[0],
            "consumer_num_whichfirm_actions": NUMFIRMS,
            "firm_state_dim": firm_state_dim,  # what are observations?
            # actions are price and wage for own firm, and capital choices
            "firm_action_dim": 3,
            "firm_num_actions": price_and_wage.shape[0],
            "government_state_dim": government_state_dim,
            "government_action_dim": 2,
            "government_num_actions": tax_choices.shape[0],
            "max_possible_consumption": float(consumption_choices.max()),
            "max_possible_hours_worked": float(work_choices.max()),
            "max_possible_wage": float(wage_choices.max()),
            "max_possible_price": float(price_choices.max()),
            # these are dims which, due to being on a large scale,
            # have to be expanded to a digit representation
            "consumer_digit_dims": global_state_digit_dims
            + [global_state_dim],  # global state + consumer budget
            # global state + firm budget (do we need capital?)
            "firm_digit_dims": global_state_digit_dims + [global_state_dim],
            # govt only has global state
            "government_digit_dims": global_state_digit_dims,
            "firm_reward_scale": 10000,
            "government_reward_scale": 100000,
            "consumer_reward_scale": 50.0,
            "firm_anneal_wages": {
                "anneal_on": True,
                "start": 22.0,
                "increase_const": float(wage_choices.max() - 22.0)
                / (episodes_to_anneal_firm),
                "decrease_const": (22.0) / episodes_to_anneal_firm,
            },
            "firm_anneal_prices": {
                "anneal_on": True,
                "start": 1000.0,
                "increase_const": float(price_choices.max() - 1000.00)
                / episodes_to_anneal_firm,
                "decrease_const": (1000.0) / episodes_to_anneal_firm,
            },
            "government_anneal_taxes": {
                "anneal_on": True,
                "start": 0.0,
                "increase_const": 1.0 / episodes_to_anneal_government,
            },
            "firm_begin_anneal_action": 0,
            "government_begin_anneal_action": government_phase1_start,
            "consumer_anneal_theta": {
                "anneal_on": True,
                "exp_decay_length_in_steps": episodes_const,
            },
            "consumer_anneal_entropy": {
                "anneal_on": True,
                "exp_decay_length_in_steps": episodes_const,
                "coef_floor": 0.1,
            },
            "firm_anneal_entropy": {
                "anneal_on": True,
                "exp_decay_length_in_steps": episodes_const,
                "coef_floor": 0.1,
            },
            "govt_anneal_entropy": {
                "anneal_on": True,
                "exp_decay_length_in_steps": episodes_const,
                "coef_floor": 0.1,
            },
            "consumer_noponzi_eta": 0.0,
            "consumer_penalty_scale": 1.0,
            "firm_noponzi_eta": 0.0,
            "firm_training_start": episodes_to_anneal_firm,
            "government_training_start": government_phase1_start
            + episodes_to_anneal_government,
            "consumer_training_start": 0,
            "government_counts_firm_reward": 0,
            "should_boost_firm_reward": False,
            "firm_reward_for_government_factor": 0.0025,
        },
        "world": {
            "maxtime": 10,
            "initial_firm_endowment": 22.0 * 1000 * NUMCONSUMERS,
            "initial_consumer_endowment": 2000,
            "initial_stocks": 0.0,
            "initial_prices": 1000.0,
            "initial_wages": 22.0,
            "interest_rate": 0.1,
            "consumer_theta": 0.01,
            "crra_param": 0.1,
            "production_alpha": "fixed_array",  # only works for exactly 10 firms, kluge
            "initial_capital": "twolevel",
            "paretoscaletheta": 4.0,
            "importer_price": 500.0,
            "importer_quantity": 100.0,
            "use_importer": 1,
        },
        "train": {
            "batch_size": 8,
            "base_seed": 1234,
            "save_dense_every": 2000,
            "save_model_every": 10000,
            "num_episodes": 500000,
            "infinite_episodes": False,
            "lr": 0.01,
            "gamma": 0.9999,
            "entropy": 0.0,
            "value_loss_weight": 1.0,
            "digit_representation_size": 10,
            "lagr_num_steps": 1,
            "boost_firm_reward_factor": 1.0,
        },
    }
    return (
        DEFAULT_CFG_DICT,
        consumption_choices,
        work_choices,
        price_and_wage,
        tax_choices,
        None,
        None,
    )


def all_agents_short_export_experiment_template(
    NUMFIRMS, NUMCONSUMERS, NUMGOVERNMENTS, episodes_const=10000
):
    consumption_choices = [
        np.array([0.0 + 1.0 * c for c in range(11)], dtype=_NP_DTYPE)
    ]
    work_choices = [
        np.array([0.0 + 20 * 13 * h for h in range(5)], dtype=_NP_DTYPE)
    ]  # specify dtype -- be consistent?

    consumption_choices = np.array(
        list(itertools.product(*consumption_choices)), dtype=_NP_DTYPE
    )
    work_choices = np.array(list(itertools.product(*work_choices)), dtype=_NP_DTYPE)

    price_choices = np.array([0.0 + 500.0 * c for c in range(6)], dtype=_NP_DTYPE)
    wage_choices = np.array([0.0, 11.0, 22.0, 33.0, 44.0], dtype=_NP_DTYPE)
    capital_choices = np.array([0.1], dtype=_NP_DTYPE)
    price_and_wage = np.array(
        list(itertools.product(price_choices, wage_choices, capital_choices)),
        dtype=_NP_DTYPE,
    )

    # government action discretization
    income_taxation_choices = np.array(
        [0.0 + 0.2 * c for c in range(6)], dtype=_NP_DTYPE
    )
    corporate_taxation_choices = np.array(
        [0.0 + 0.2 * c for c in range(6)], dtype=_NP_DTYPE
    )
    tax_choices = np.array(
        list(itertools.product(income_taxation_choices, corporate_taxation_choices)),
        dtype=_NP_DTYPE,
    )
    global_state_dim = (
        NUMFIRMS  # prices
        + NUMFIRMS  # wages
        + NUMFIRMS  # stocks
        + NUMFIRMS  # was good overdemanded
        + 2 * NUMGOVERNMENTS  # tax rates
        + 1
    )  # time

    global_state_digit_dims = list(
        range(2 * NUMFIRMS, 3 * NUMFIRMS)
    )  # stocks are the only global state var that can get huge
    consumer_state_dim = (
        global_state_dim + 1 + 1
    )  # budget  # theta, the disutility of work

    firm_state_dim = (
        global_state_dim
        + 1  # budget
        + 1  # capital
        + 1  # production alpha
        + NUMFIRMS  # onehot specifying which firm
    )

    episodes_to_anneal_firm = 30000
    episodes_to_anneal_government = 30000
    government_phase1_start = 30000
    government_state_dim = global_state_dim
    DEFAULT_CFG_DICT = {
        # actions_array key will be added below
        "agents": {
            "num_consumers": NUMCONSUMERS,
            "num_firms": NUMFIRMS,
            "num_governments": NUMGOVERNMENTS,
            "global_state_dim": global_state_dim,
            "consumer_state_dim": consumer_state_dim,
            # action vectors are how much consume from each firm,
            # how much to work, and which firm to choose
            "consumer_action_dim": NUMFIRMS + 1 + 1,
            "consumer_num_consume_actions": consumption_choices.shape[0],
            "consumer_num_work_actions": work_choices.shape[0],
            "consumer_num_whichfirm_actions": NUMFIRMS,
            "firm_state_dim": firm_state_dim,  # what are observations?
            # actions are price and wage for own firm, and capital choices
            "firm_action_dim": 3,
            "firm_num_actions": price_and_wage.shape[0],
            "government_state_dim": government_state_dim,
            "government_action_dim": 2,
            "government_num_actions": tax_choices.shape[0],
            "max_possible_consumption": float(consumption_choices.max()),
            "max_possible_hours_worked": float(work_choices.max()),
            "max_possible_wage": float(wage_choices.max()),
            "max_possible_price": float(price_choices.max()),
            # these are dims which, due to being on a large scale,
            # have to be expanded to a digit representation
            "consumer_digit_dims": global_state_digit_dims
            + [global_state_dim],  # global state + consumer budget
            "firm_digit_dims": global_state_digit_dims
            + [global_state_dim],  # global state + firm budget (do we need capital?)
            # govt only has global state
            "government_digit_dims": global_state_digit_dims,
            "firm_reward_scale": 10000,
            "government_reward_scale": 100000,
            "consumer_reward_scale": 50.0,
            "firm_anneal_wages": {
                "anneal_on": True,
                "start": 22.0,
                "increase_const": float(wage_choices.max() - 22.0)
                / (episodes_to_anneal_firm),
                "decrease_const": (22.0) / episodes_to_anneal_firm,
            },
            "firm_anneal_prices": {
                "anneal_on": True,
                "start": 1000.0,
                "increase_const": float(price_choices.max() - 1000.00)
                / episodes_to_anneal_firm,
                "decrease_const": (1000.0) / episodes_to_anneal_firm,
            },
            "government_anneal_taxes": {
                "anneal_on": True,
                "start": 0.0,
                "increase_const": 1.0 / episodes_to_anneal_government,
            },
            "firm_begin_anneal_action": 0,
            "government_begin_anneal_action": government_phase1_start,
            "consumer_anneal_theta": {
                "anneal_on": True,
                "exp_decay_length_in_steps": episodes_const,
            },
            "consumer_anneal_entropy": {
                "anneal_on": True,
                "exp_decay_length_in_steps": episodes_const,
                "coef_floor": 0.1,
            },
            "firm_anneal_entropy": {
                "anneal_on": True,
                "exp_decay_length_in_steps": episodes_const,
                "coef_floor": 0.1,
            },
            "govt_anneal_entropy": {
                "anneal_on": True,
                "exp_decay_length_in_steps": episodes_const,
                "coef_floor": 0.1,
            },
            "consumer_noponzi_eta": 0.0,
            "consumer_penalty_scale": 1.0,
            "firm_noponzi_eta": 0.0,
            "firm_training_start": episodes_to_anneal_firm,
            "government_training_start": government_phase1_start
            + episodes_to_anneal_government,
            "consumer_training_start": 0,
            "government_counts_firm_reward": 0,
            "should_boost_firm_reward": False,
            "firm_reward_for_government_factor": 0.0025,
        },
        "world": {
            "maxtime": 10,
            "initial_firm_endowment": 22.0 * 1000 * NUMCONSUMERS,
            "initial_consumer_endowment": 2000,
            "initial_stocks": 0.0,
            "initial_prices": 1000.0,
            "initial_wages": 22.0,
            "interest_rate": 0.1,
            "consumer_theta": 0.01,
            "crra_param": 0.1,
            "production_alpha": "fixed_array",  # only works for exactly 10 firms, kluge
            "initial_capital": "twolevel",
            "paretoscaletheta": 4.0,
            "importer_price": 500.0,
            "importer_quantity": 100.0,
            "use_importer": 1,
        },
        "train": {
            "batch_size": 8,
            "base_seed": 1234,
            "save_dense_every": 2000,
            "save_model_every": 10000,
            "num_episodes": 200000,
            "infinite_episodes": False,
            "lr": 0.01,
            "gamma": 0.9999,
            "entropy": 0.0,
            "value_loss_weight": 1.0,
            "digit_representation_size": 10,
            "lagr_num_steps": 1,
            "boost_firm_reward_factor": 1.0,
        },
    }
    return (
        DEFAULT_CFG_DICT,
        consumption_choices,
        work_choices,
        price_and_wage,
        tax_choices,
        None,
        None,
    )


def very_short_test_template(
    NUMFIRMS, NUMCONSUMERS, NUMGOVERNMENTS, episodes_const=30000
):
    consumption_choices = [
        np.array([0.0 + 1.0 * c for c in range(11)], dtype=_NP_DTYPE)
    ]
    work_choices = [
        np.array([0.0 + 20 * 13 * h for h in range(5)], dtype=_NP_DTYPE)
    ]  # specify dtype -- be consistent?

    consumption_choices = np.array(
        list(itertools.product(*consumption_choices)), dtype=_NP_DTYPE
    )
    work_choices = np.array(list(itertools.product(*work_choices)), dtype=_NP_DTYPE)

    price_choices = np.array([0.0 + 500.0 * c for c in range(6)], dtype=_NP_DTYPE)
    wage_choices = np.array([0.0, 11.0, 22.0, 33.0, 44.0], dtype=_NP_DTYPE)
    capital_choices = np.array([0.1], dtype=_NP_DTYPE)
    price_and_wage = np.array(
        list(itertools.product(price_choices, wage_choices, capital_choices)),
        dtype=_NP_DTYPE,
    )

    # government action discretization
    income_taxation_choices = np.array(
        [0.0 + 0.2 * c for c in range(6)], dtype=_NP_DTYPE
    )
    corporate_taxation_choices = np.array(
        [0.0 + 0.2 * c for c in range(6)], dtype=_NP_DTYPE
    )
    tax_choices = np.array(
        list(itertools.product(income_taxation_choices, corporate_taxation_choices)),
        dtype=_NP_DTYPE,
    )
    global_state_dim = (
        NUMFIRMS  # prices
        + NUMFIRMS  # wages
        + NUMFIRMS  # stocks
        + NUMFIRMS  # was good overdemanded
        + 2 * NUMGOVERNMENTS  # tax rates
        + 1
    )  # time

    global_state_digit_dims = list(
        range(2 * NUMFIRMS, 3 * NUMFIRMS)
    )  # stocks are the only global state var that can get huge
    consumer_state_dim = (
        global_state_dim + 1 + 1
    )  # budget  # theta, the disutility of work

    firm_state_dim = (
        global_state_dim
        + 1  # budget
        + 1  # capital
        + 1  # production alpha
        + NUMFIRMS  # onehot specifying which firm
    )

    episodes_to_anneal_firm = 10
    episodes_to_anneal_government = 10
    government_phase1_start = 10
    government_state_dim = global_state_dim
    DEFAULT_CFG_DICT = {
        # actions_array key will be added below
        "agents": {
            "num_consumers": NUMCONSUMERS,
            "num_firms": NUMFIRMS,
            "num_governments": NUMGOVERNMENTS,
            "global_state_dim": global_state_dim,
            "consumer_state_dim": consumer_state_dim,
            # action vectors are how much consume from each firm,
            # how much to work, and which firm to choose
            "consumer_action_dim": NUMFIRMS + 1 + 1,
            "consumer_num_consume_actions": consumption_choices.shape[0],
            "consumer_num_work_actions": work_choices.shape[0],
            "consumer_num_whichfirm_actions": NUMFIRMS,
            "firm_state_dim": firm_state_dim,  # what are observations?
            # actions are price and wage for own firm, and capital choices
            "firm_action_dim": 3,
            "firm_num_actions": price_and_wage.shape[0],
            "government_state_dim": government_state_dim,
            "government_action_dim": 2,
            "government_num_actions": tax_choices.shape[0],
            "max_possible_consumption": float(consumption_choices.max()),
            "max_possible_hours_worked": float(work_choices.max()),
            "max_possible_wage": float(wage_choices.max()),
            "max_possible_price": float(price_choices.max()),
            # these are dims which, due to being on a large scale,
            # have to be expanded to a digit representation
            "consumer_digit_dims": global_state_digit_dims
            + [global_state_dim],  # global state + consumer budget
            "firm_digit_dims": global_state_digit_dims
            + [global_state_dim],  # global state + firm budget (do we need capital?)
            # govt only has global state
            "government_digit_dims": global_state_digit_dims,
            "firm_reward_scale": 10000,
            "government_reward_scale": 100000,
            "consumer_reward_scale": 50.0,
            "firm_anneal_wages": {
                "anneal_on": True,
                "start": 22.0,
                "increase_const": float(wage_choices.max() - 22.0)
                / (episodes_to_anneal_firm),
                "decrease_const": (22.0) / episodes_to_anneal_firm,
            },
            "firm_anneal_prices": {
                "anneal_on": True,
                "start": 1000.0,
                "increase_const": float(price_choices.max() - 1000.00)
                / episodes_to_anneal_firm,
                "decrease_const": (1000.0) / episodes_to_anneal_firm,
            },
            "government_anneal_taxes": {
                "anneal_on": True,
                "start": 0.0,
                "increase_const": 1.0 / episodes_to_anneal_government,
            },
            "firm_begin_anneal_action": 0,
            "government_begin_anneal_action": government_phase1_start,
            "consumer_anneal_theta": {
                "anneal_on": True,
                "exp_decay_length_in_steps": episodes_const,
            },
            "consumer_anneal_entropy": {
                "anneal_on": True,
                "exp_decay_length_in_steps": episodes_const,
                "coef_floor": 0.1,
            },
            "firm_anneal_entropy": {
                "anneal_on": True,
                "exp_decay_length_in_steps": episodes_const,
                "coef_floor": 0.1,
            },
            "govt_anneal_entropy": {
                "anneal_on": True,
                "exp_decay_length_in_steps": episodes_const,
                "coef_floor": 0.1,
            },
            "consumer_noponzi_eta": 0.0,
            "consumer_penalty_scale": 1.0,
            "firm_noponzi_eta": 0.0,
            "firm_training_start": episodes_to_anneal_firm,
            "government_training_start": government_phase1_start
            + episodes_to_anneal_government,
            "consumer_training_start": 0,
            "government_counts_firm_reward": 0,
            "should_boost_firm_reward": False,
            "firm_reward_for_government_factor": 0.0025,
            "train_firms_every": 2,
            "train_consumers_every": 1,
            "train_government_every": 5,
        },
        "world": {
            "maxtime": 10,
            "initial_firm_endowment": 22.0 * 1000 * NUMCONSUMERS,
            "initial_consumer_endowment": 2000,
            "initial_stocks": 0.0,
            "initial_prices": 1000.0,
            "initial_wages": 22.0,
            "interest_rate": 0.1,
            "consumer_theta": 0.01,
            "crra_param": 0.1,
            "production_alpha": "fixed_array",  # only works for exactly 10 firms, kluge
            "initial_capital": "twolevel",
            "paretoscaletheta": 4.0,
            "importer_price": 500.0,
            "importer_quantity": 100.0,
            "use_importer": 1,
        },
        "train": {
            "batch_size": 8,
            "base_seed": 1234,
            "save_dense_every": 2000,
            "save_model_every": 10000,
            "num_episodes": 100,
            "infinite_episodes": False,
            "lr": 0.01,
            "gamma": 0.9999,
            "entropy": 0.0,
            "value_loss_weight": 1.0,
            "digit_representation_size": 10,
            "lagr_num_steps": 1,
            "boost_firm_reward_factor": 1.0,
        },
    }
    return (
        DEFAULT_CFG_DICT,
        consumption_choices,
        work_choices,
        price_and_wage,
        tax_choices,
        None,
        None,
    )


def global_state_scaling_factors(cfg_dict):
    max_wage = cfg_dict["agents"]["max_possible_wage"]
    max_price = cfg_dict["agents"]["max_possible_price"]
    num_firms = cfg_dict["agents"]["num_firms"]
    num_governments = cfg_dict["agents"]["num_governments"]
    maxtime = cfg_dict["world"]["maxtime"]

    digit_size = cfg_dict["train"]["digit_representation_size"]

    return torch.tensor(
        # prices, wages, stocks, overdemanded
        ([max_price] * num_firms)
        + ([max_wage] * num_firms)
        + ([1.0] * num_firms * digit_size)  # stocks are expanded to digit form
        + ([1.0] * num_firms)
        + ([1.0] * (2 * num_governments))
        + [maxtime]
    )


def consumer_state_scaling_factors(cfg_dict):
    global_state_scales = global_state_scaling_factors(cfg_dict)
    digit_size = cfg_dict["train"]["digit_representation_size"]
    consumer_scales = torch.tensor(
        ([1.0] * digit_size) + [cfg_dict["world"]["consumer_theta"]]
    )
    return torch.cat((global_state_scales, consumer_scales)).cuda()


def firm_state_scaling_factors(cfg_dict):
    num_firms = cfg_dict["agents"]["num_firms"]
    global_state_scales = global_state_scaling_factors(cfg_dict)
    digit_size = cfg_dict["train"]["digit_representation_size"]
    # budget, capital, alpha, one-hot
    firm_scales = torch.tensor(
        ([1.0] * digit_size) + [10000.0, 1.0] + ([1.0] * num_firms)
    )
    return torch.cat((global_state_scales, firm_scales)).cuda()


def govt_state_scaling_factors(cfg_dict):
    return global_state_scaling_factors(cfg_dict).cuda()
