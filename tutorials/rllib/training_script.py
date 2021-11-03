# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import argparse
import logging
import os
import sys
import time

import ray
from utils import remote, saving
import tf_models
import yaml
from env_wrapper import RLlibEnvWrapper
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.logger import NoopLogger, pretty_print

ray.init(log_to_driver=False)

logging.basicConfig(stream=sys.stdout, format="%(asctime)s %(message)s")
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)


def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir", type=str, help="Path to the directory for this run."
    )

    args = parser.parse_args()
    run_directory = args.run_dir

    config_path = os.path.join(args.run_dir, "config.yaml")
    assert os.path.isdir(args.run_dir)
    assert os.path.isfile(config_path)

    with open(config_path, "r") as f:
        run_configuration = yaml.safe_load(f)

    return run_directory, run_configuration


def build_trainer(run_configuration):
    """Finalize the trainer config by combining the sub-configs."""
    trainer_config = run_configuration.get("trainer")

    # === Env ===
    env_config = {
        "env_config_dict": run_configuration.get("env"),
        "num_envs_per_worker": trainer_config.get("num_envs_per_worker"),
    }

    # === Seed ===
    if trainer_config["seed"] is None:
        try:
            start_seed = int(run_configuration["metadata"]["launch_time"])
        except KeyError:
            start_seed = int(time.time())
    else:
        start_seed = int(trainer_config["seed"])

    final_seed = int(start_seed % (2 ** 16)) * 1000
    logger.info("seed (final): %s", final_seed)

    # === Multiagent Policies ===
    dummy_env = RLlibEnvWrapper(env_config)

    # Policy tuples for agent/planner policy types
    agent_policy_tuple = (
        None,
        dummy_env.observation_space,
        dummy_env.action_space,
        run_configuration.get("agent_policy"),
    )
    planner_policy_tuple = (
        None,
        dummy_env.observation_space_pl,
        dummy_env.action_space_pl,
        run_configuration.get("planner_policy"),
    )

    policies = {"a": agent_policy_tuple, "p": planner_policy_tuple}

    def policy_mapping_fun(i):
        if str(i).isdigit() or i == "a":
            return "a"
        return "p"

    # Which policies to train
    if run_configuration["general"]["train_planner"]:
        policies_to_train = ["a", "p"]
    else:
        policies_to_train = ["a"]

    # === Finalize and create ===
    trainer_config.update(
        {
            "env_config": env_config,
            "seed": final_seed,
            "multiagent": {
                "policies": policies,
                "policies_to_train": policies_to_train,
                "policy_mapping_fn": policy_mapping_fun,
            },
            "metrics_smoothing_episodes": trainer_config.get("num_workers")
            * trainer_config.get("num_envs_per_worker"),
        }
    )

    def logger_creator(config):
        return NoopLogger({}, "/tmp")

    ppo_trainer = PPOTrainer(
        env=RLlibEnvWrapper, config=trainer_config, logger_creator=logger_creator
    )

    return ppo_trainer


def set_up_dirs_and_maybe_restore(run_directory, run_configuration, trainer_obj):
    # === Set up Logging & Saving, or Restore ===
    # All model parameters are always specified in the settings YAML.
    # We do NOT overwrite / reload settings from the previous checkpoint dir.
    # 1. For new runs, the only object that will be loaded from the checkpoint dir
    #    are model weights.
    # 2. For crashed and restarted runs, load_snapshot will reload the full state of
    #    the Trainer(s), including metadata, optimizer, and models.
    (
        dense_log_directory,
        ckpt_directory,
        restore_from_crashed_run,
    ) = saving.fill_out_run_dir(run_directory)

    # If this is a starting from a crashed run, restore the last trainer snapshot
    if restore_from_crashed_run:
        logger.info(
            "ckpt_dir already exists! Planning to restore using latest snapshot from "
            "earlier (crashed) run with the same ckpt_dir %s",
            ckpt_directory,
        )

        at_loads_a_ok = saving.load_snapshot(
            trainer_obj, run_directory, load_latest=True
        )

        # at this point, we need at least one good ckpt restored
        if not at_loads_a_ok:
            logger.fatal(
                "restore_from_crashed_run -> restore_run_dir %s, but no good ckpts "
                "found/loaded!",
                run_directory,
            )
            sys.exit()

        # === Trainer-specific counters ===
        training_step_last_ckpt = (
            int(trainer_obj._timesteps_total) if trainer_obj._timesteps_total else 0
        )
        epis_last_ckpt = (
            int(trainer_obj._episodes_total) if trainer_obj._episodes_total else 0
        )

    else:
        logger.info("Not restoring trainer...")
        # === Trainer-specific counters ===
        training_step_last_ckpt = 0
        epis_last_ckpt = 0

        # For new runs, load only tf checkpoint weights
        starting_weights_path_agents = run_configuration["general"].get(
            "restore_tf_weights_agents", ""
        )
        if starting_weights_path_agents:
            logger.info("Restoring agents TF weights...")
            saving.load_tf_model_weights(trainer_obj, starting_weights_path_agents)
        else:
            logger.info("Starting with fresh agent TF weights.")

        starting_weights_path_planner = run_configuration["general"].get(
            "restore_tf_weights_planner", ""
        )
        if starting_weights_path_planner:
            logger.info("Restoring planner TF weights...")
            saving.load_tf_model_weights(trainer_obj, starting_weights_path_planner)
        else:
            logger.info("Starting with fresh planner TF weights.")

    return (
        dense_log_directory,
        ckpt_directory,
        restore_from_crashed_run,
        training_step_last_ckpt,
        epis_last_ckpt,
    )


def maybe_sync_saez_buffer(trainer_obj, result_dict, run_configuration):
    if result_dict["episodes_this_iter"] == 0:
        return

    # This logic just detects if we're using the Saez formula
    sync_saez = False
    for component in run_configuration["env"]["components"]:
        assert isinstance(component, dict)
        c_name = list(component.keys())[0]
        c_kwargs = list(component.values())[0]
        if c_name in ["PeriodicBracketTax"]:
            tax_model = c_kwargs.get("tax_model", "")
            if tax_model == "saez":
                sync_saez = True
                break

    # Do the actual syncing
    if sync_saez:
        remote.accumulate_and_broadcast_saez_buffers(trainer_obj)


def maybe_store_dense_log(
    trainer_obj, result_dict, dense_log_freq, dense_log_directory
):
    if result_dict["episodes_this_iter"] > 0 and dense_log_freq > 0:
        episodes_per_replica = (
            result_dict["episodes_total"] // result_dict["episodes_this_iter"]
        )
        if episodes_per_replica == 1 or (episodes_per_replica % dense_log_freq) == 0:
            log_dir = os.path.join(
                dense_log_directory,
                "logs_{:016d}".format(result_dict["timesteps_total"]),
            )
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
            saving.write_dense_logs(trainer_obj, log_dir)
            logger.info(">> Wrote dense logs to: %s", log_dir)


def maybe_save(trainer_obj, result_dict, ckpt_freq, ckpt_directory, trainer_step_last_ckpt):
    global_step = result_dict["timesteps_total"]

    # Check if saving this iteration
    if (
        result_dict["episodes_this_iter"] > 0
    ):  # Don't save if midway through an episode.

        if ckpt_freq > 0:
            if global_step - trainer_step_last_ckpt >= ckpt_freq:
                saving.save_snapshot(trainer_obj, ckpt_directory, suffix="")
                saving.save_tf_model_weights(
                    trainer_obj, ckpt_directory, global_step, suffix="agent"
                )
                saving.save_tf_model_weights(
                    trainer_obj, ckpt_directory, global_step, suffix="planner"
                )

                trainer_step_last_ckpt = int(global_step)

                logger.info("Checkpoint saved @ step %d", global_step)

    return trainer_step_last_ckpt


if __name__ == "__main__":

    # ===================
    # === Start setup ===
    # ===================

    # Process the args
    run_dir, run_config = process_args()

    # Create a trainer object
    trainer = build_trainer(run_config)

    # Set up directories for logging and saving. Restore if this has already been
    # done (indicating that we're restarting a crashed run). Or, if appropriate,
    # load in starting model weights for the agent and/or planner.
    (
        dense_log_dir,
        ckpt_dir,
        restore_from_crashed_run,
        step_last_ckpt,
        num_parallel_episodes_done,
    ) = set_up_dirs_and_maybe_restore(run_dir, run_config, trainer)

    # ======================
    # === Start training ===
    # ======================
    dense_log_frequency = run_config["env"].get("dense_log_frequency", 0)
    ckpt_frequency = run_config["general"].get("ckpt_frequency_steps", 0)
    global_step = int(step_last_ckpt)

    while num_parallel_episodes_done < run_config["general"]["episodes"]:

        # Training
        result = trainer.train()

        # === Counters++ ===
        num_parallel_episodes_done = result["episodes_total"]
        global_step = result["timesteps_total"]
        curr_iter = result["training_iteration"]

        logger.info(
            "Iter %d: steps this-iter %d total %d -> %d/%d episodes done",
            curr_iter,
            result["timesteps_this_iter"],
            global_step,
            num_parallel_episodes_done,
            run_config["general"]["episodes"],
        )

        if curr_iter == 1 or result["episodes_this_iter"] > 0:
            logger.info(pretty_print(result))

        # === Saez logic ===
        maybe_sync_saez_buffer(trainer, result, run_config)

        # === Dense logging ===
        maybe_store_dense_log(trainer, result, dense_log_frequency, dense_log_dir)

        # === Saving ===
        step_last_ckpt = maybe_save(
            trainer, result, ckpt_frequency, ckpt_dir, step_last_ckpt
        )

    # Finish up
    logger.info("Completing! Saving final snapshot...\n\n")
    saving.save_snapshot(trainer, ckpt_dir)
    saving.save_tf_model_weights(trainer, ckpt_dir, global_step, suffix="agent")
    saving.save_tf_model_weights(trainer, ckpt_dir, global_step, suffix="planner")
    logger.info("Final snapshot saved! All done.")

    ray.shutdown()  # shutdown Ray after use
