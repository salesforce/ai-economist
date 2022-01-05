# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import hashlib
import itertools
import json
import os
import pickle
import struct
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

# defaults
_NUM_FIRMS = 10


def _bigint_from_bytes(num_bytes):
    """
    See https://github.com/openai/gym/blob/master/gym/utils/seeding.py.
    """
    sizeof_int = 4
    padding = sizeof_int - len(num_bytes) % sizeof_int
    num_bytes += b"\0" * padding
    int_count = int(len(num_bytes) / sizeof_int)
    unpacked = struct.unpack("{}I".format(int_count), num_bytes)
    accum = 0
    for i, val in enumerate(unpacked):
        accum += 2 ** (sizeof_int * 8 * i) * val
    return accum


def seed_from_base_seed(base_seed):
    """
    Hash base seed to reduce correlation.
    """
    max_bytes = 4
    hash_func = hashlib.sha512(str(base_seed).encode("utf8")).digest()

    return _bigint_from_bytes(hash_func[:max_bytes])


def hash_from_dict(d):
    d_copy = deepcopy(d)
    del (d_copy["train"])["base_seed"]
    d_string = json.dumps(d_copy, sort_keys=True)
    return int(hashlib.sha256(d_string.encode("utf8")).hexdigest()[:8], 16)


def cfg_dict_from_yaml(
    hparams_path,
    consumption_choices,
    work_choices,
    price_and_wage,
    tax_choices,
    group_name=None,
):
    with open(hparams_path) as f:
        d = yaml.safe_load(f)

    if group_name is not None:
        d["metadata"]["group_name"] = group_name
    d["metadata"]["hparamhash"] = hash_from_dict(d)
    d["agents"][
        "consumer_consumption_actions_array"
    ] = consumption_choices  # Note: hardcoded
    d["agents"]["consumer_work_actions_array"] = work_choices  # Note: hardcoded
    d["agents"]["firm_actions_array"] = price_and_wage  # Note: hardcoded
    d["agents"]["government_actions_array"] = tax_choices
    d["train"]["save_dir"] = str(hparams_path.absolute().parent)
    d["train"]["seed"] = seed_from_base_seed(d["train"]["base_seed"])
    return d


def run_experiment_batch_parallel(
    experiment_dir,
    consumption_choices,
    work_choices,
    price_and_wage,
    tax_choices,
    group_name=None,
    consumers_only=False,
    no_firms=False,
    default_firm_action=None,
    default_government_action=None,
):
    hparams_path = Path(experiment_dir) / Path("hparams.yaml")
    hparams_dict = cfg_dict_from_yaml(
        hparams_path,
        consumption_choices,
        work_choices,
        price_and_wage,
        tax_choices,
        group_name=group_name,
    )
    print(f"hparams_dict {hparams_dict}")
    # import this here so rest of file still imports without cuda installed
    from rbc.cuda_manager import ConsumerFirmRunManagerBatchParallel

    if consumers_only:
        m = ConsumerFirmRunManagerBatchParallel(
            hparams_dict,
            freeze_firms=default_firm_action,
            freeze_govt=default_government_action,
        )
    elif no_firms:
        m = ConsumerFirmRunManagerBatchParallel(
            hparams_dict,
            freeze_firms=default_firm_action,
        )
    else:
        m = ConsumerFirmRunManagerBatchParallel(hparams_dict)
    m.train()


def compare_global_states_within_type(states, global_state_size):
    # every agent within a batch should have the same global state
    first_agent_global = states[:, :, :1, :global_state_size]
    all_agents_global = states[:, :, :, :global_state_size]
    return np.isclose(all_agents_global, first_agent_global).all()


def compare_global_states_across_types(
    consumer_states, firm_states, government_states, global_state_size
):
    first_agent_global = consumer_states[:, :, :1, :global_state_size]
    return (
        np.isclose(firm_states[:, :, :, :global_state_size], first_agent_global).all(),
        np.isclose(
            government_states[:, :, :, :global_state_size], first_agent_global
        ).all(),
        np.isclose(
            consumer_states[:, :, :, :global_state_size], first_agent_global
        ).all(),
    )


def check_no_negative_stocks(state, stock_offset, stock_size):
    stocks = state[:, :, :, stock_offset : (stock_offset + stock_size)]
    return (stocks >= -1.0e-3).all()


train_param_sweeps = {
    "lr": [0.005, 0.001],
    "entropy": [0.01],
    "base_seed": [2596],
    "batch_size": [64],
    "clip_grad_norm": [1.0, 2.0, 5.0],
}

# Other param sweeps
agent_param_sweeps = {
    # "consumer_noponzi_eta": [0.1,0.05]
}

world_param_sweeps = {
    # "interest_rate": [0.02, 0.0]
}


def add_all(d, keys_list, target_val):
    for k in keys_list:
        d[k] = target_val


def sweep_cfg_generator(
    base_cfg,
    tr_param_sweeps=None,
    ag_param_sweeps=None,
    wld_param_sweeps=None,
    seed_from_timestamp=False,
    group_name=None,
):
    # train_param_sweeps
    if tr_param_sweeps is None:
        tr_param_sweeps = {}
    # agent_param_sweeps
    if ag_param_sweeps is None:
        ag_param_sweeps = {}
    # world_param_sweeps
    if wld_param_sweeps is None:
        wld_param_sweeps = {}

    assert isinstance(tr_param_sweeps, dict)
    assert isinstance(ag_param_sweeps, dict)
    assert isinstance(wld_param_sweeps, dict)

    key_dict = {}  # tells which key goes to which dict, e.g. "lr" -> "train", etc.
    if len(tr_param_sweeps) > 0:
        train_k, train_v = zip(*tr_param_sweeps.items())
    else:
        train_k, train_v = (), ()
    add_all(key_dict, train_k, "train")
    if len(ag_param_sweeps) > 0:
        agent_k, agent_v = zip(*ag_param_sweeps.items())
    else:
        agent_k, agent_v = (), ()
    add_all(key_dict, agent_k, "agents")
    if len(wld_param_sweeps) > 0:
        world_k, world_v = zip(*wld_param_sweeps.items())
    else:
        world_k, world_v = (), ()
    add_all(key_dict, world_k, "world")

    k = train_k + agent_k + world_k
    v = train_v + agent_v + world_v

    # have a "reverse lookup" dictionary for each key name
    for combination in itertools.product(*v):
        values_to_substitute = dict(zip(k, combination))
        out = deepcopy(base_cfg)
        for key, value in values_to_substitute.items():
            out[key_dict[key]][key] = value
        if seed_from_timestamp:
            int_timestamp = int(
                time.time() * 1000
            )  # time.time() returns float, multiply 1000 for higher resolution
            out["train"]["base_seed"] += int_timestamp
        if group_name is not None:
            out["metadata"]["group"] = group_name
        yield out


def create_job_dir(experiment_dir, job_name_base, cfg=None, action_arrays=None):
    unique_id = time.time()
    dirname = f"{job_name_base}-{unique_id}".replace(".", "-")
    dir_path = Path(experiment_dir) / Path(dirname)
    os.makedirs(str(dir_path), exist_ok=True)
    cfg["metadata"]["dirname"] = dirname
    cfg["metadata"]["group"] = str(Path(experiment_dir).name)
    with open(dir_path / Path("hparams.yaml"), "w") as f:
        f.write(yaml.dump(cfg))

    if action_arrays is not None:
        with open(dir_path / Path("action_arrays.pickle"), "wb") as f:
            pickle.dump(action_arrays, f)
