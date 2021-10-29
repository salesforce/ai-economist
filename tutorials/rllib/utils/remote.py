# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np


def remote_env_fun(trainer, env_function):
    """
    Create a dictionary with the following mapping:
        result[env_wrapper.env_id] = env_function(env)
    where each entry in the dictionary comes from one of the active envs in the trainer.
    env_function must be a function that takes an environment as its single argument
    """

    nested_env_ids_and_results = trainer.workers.foreach_worker(
        lambda w: [(env.env_id, env_function(env)) for env in w.async_env.envs]
    )
    nested_env_ids_and_results = nested_env_ids_and_results[
        1:
    ]  # Ignore the local worker

    # Store them first this way in case they don't come out sorted
    # (gets sorted by env_id before being returned)
    result = {}

    for worker_stuff in nested_env_ids_and_results:
        for env_id, output in worker_stuff:
            result[env_id] = output
    return result


def get_trainer_envs(trainer):
    return remote_env_fun(trainer, lambda env: env)


def collect_stored_rollouts(trainer):
    aggregate_rollouts = {}

    rollout_dict = remote_env_fun(trainer, lambda e: e.rollout)
    n_envs = len(rollout_dict)

    for env_id, env_rollout in rollout_dict.items():
        for k, v in env_rollout.items():
            if k not in aggregate_rollouts:
                sz = v.shape
                sz = [sz[0], n_envs] + sz[1:]
                aggregate_rollouts[k] = np.zeros(sz)
            aggregate_rollouts[k][:, env_id] = v

    return aggregate_rollouts


def accumulate_and_broadcast_saez_buffers(trainer):
    component_name = "PeriodicBracketTax"

    def extract_local_saez_buffers(env_wrapper):
        return env_wrapper.env.get_component(component_name).get_local_saez_buffer()

    replica_buffers = remote_env_fun(trainer, extract_local_saez_buffers)

    global_buffer = []
    for local_buffer in replica_buffers.values():
        global_buffer += local_buffer

    def set_global_buffer(env_wrapper):
        env_wrapper.env.get_component(component_name).set_global_saez_buffer(
            global_buffer
        )

    _ = remote_env_fun(trainer, set_global_buffer)
