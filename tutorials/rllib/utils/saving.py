# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import logging
import os
import pickle
import shutil
import sys

import yaml
from ai_economist import foundation
from .remote import remote_env_fun

logging.basicConfig(format="%(asctime)s %(message)s")
logger = logging.getLogger("utils")
logger.setLevel(logging.DEBUG)


sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(sys.modules[__name__].__file__), "../../..")
    )
)


def fill_out_run_dir(run_dir):
    dense_log_dir = os.path.join(run_dir, "dense_logs")
    ckpt_dir = os.path.join(run_dir, "ckpts")

    for sub_dir in [dense_log_dir, ckpt_dir]:
        os.makedirs(sub_dir, exist_ok=True)

    latest_filepath = os.path.join(ckpt_dir, "latest_checkpoint.pkl")
    restore = bool(os.path.isfile(latest_filepath))

    return dense_log_dir, ckpt_dir, restore


def make_run_dir_path(args, log_group):
    assert os.path.isdir(args.top_experiment_dir)
    assert args.launch_time

    project_dir = os.path.join(args.top_experiment_dir, args.project)
    if not os.path.isdir(project_dir):
        try:
            os.makedirs(project_dir)
        except FileExistsError:
            pass

    experiment_dir = os.path.join(project_dir, args.experiment)
    if not os.path.isdir(experiment_dir):
        try:
            os.makedirs(experiment_dir)
        except FileExistsError:
            pass

    group_dir = os.path.join(experiment_dir, log_group)
    if not os.path.isdir(group_dir):
        try:
            os.makedirs(group_dir)
        except FileExistsError:
            pass

    run_dir = os.path.join(
        group_dir,
        "launch_time_{}".format(args.launch_time),
        "run_id_{}".format(args.unique_run_id),
    )
    debug_dir = os.path.join(run_dir, "debug")
    dense_log_dir = os.path.join(run_dir, "dense_logs")
    ckpt_dir = os.path.join(run_dir, "ckpts")
    latest_filepath = os.path.join(ckpt_dir, "latest_checkpoint.pkl")

    if not os.path.isdir(run_dir):
        restore = False
        os.makedirs(run_dir)
        os.makedirs(debug_dir)
        os.makedirs(dense_log_dir)
        os.makedirs(ckpt_dir)

    elif os.path.isfile(latest_filepath):
        restore = True

    else:
        restore = False

    return run_dir, debug_dir, dense_log_dir, ckpt_dir, restore


def write_dense_logs(trainer, log_directory, suffix=""):
    def save_log(env_wrapper):
        if 0 <= env_wrapper.env_id < 4:
            my_path = os.path.join(
                log_directory,
                "env{:03d}{}.lz4".format(
                    env_wrapper.env_id, "." + suffix if suffix != "" else ""
                ),
            )
            foundation.utils.save_episode_log(env_wrapper.env, my_path)

    remote_env_fun(trainer, save_log)


def save_tf_model_weights(trainer, ckpt_dir, global_step, suffix=""):
    if suffix == "agent":
        w = trainer.get_weights(["a"])
        pol = trainer.get_policy("a")
        model_w_array = pol._sess.run(pol.model.variables())
    elif suffix == "planner":
        w = trainer.get_weights(["p"])
        pol = trainer.get_policy("p")
        model_w_array = pol._sess.run(pol.model.variables())
    else:
        raise NotImplementedError

    fn = os.path.join(
        ckpt_dir, "{}.tf.weights.global-step-{}".format(suffix, global_step)
    )
    with open(fn, "wb") as f:
        pickle.dump(w, f)

    fn = os.path.join(
        ckpt_dir,
        "{}.policy-model-weight-array.global-step-{}".format(suffix, global_step),
    )
    with open(fn, "wb") as f:
        pickle.dump(model_w_array, f)

    logger.info("Saved TF weights @ %s", fn)


def load_tf_model_weights(trainer, ckpt):
    assert os.path.isfile(ckpt)
    with open(ckpt, "rb") as f:
        weights = pickle.load(f)
        trainer.set_weights(weights)
    logger.info("loaded tf model weights:\n\t%s\n", ckpt)


def save_snapshot(trainer, ckpt_dir, suffix=""):
    # Create a new trainer snapshot
    filepath = trainer.save(ckpt_dir)
    filepath_metadata = filepath + ".tune_metadata"
    # Copy this to a standardized name (to only keep the latest)
    latest_filepath = os.path.join(
        ckpt_dir, "latest_checkpoint{}.pkl".format("." + suffix if suffix != "" else "")
    )
    latest_filepath_metadata = latest_filepath + ".tune_metadata"
    shutil.copy(filepath, latest_filepath)
    shutil.copy(filepath_metadata, latest_filepath_metadata)
    # Get rid of the timestamped copy to prevent accumulating too many large files
    os.remove(filepath)
    os.remove(filepath_metadata)

    # Also take snapshots of each environment object
    remote_env_fun(trainer, lambda env_wrapper: env_wrapper.save_game_object(ckpt_dir))

    logger.info("Saved Trainer snapshot + Env object @ %s", latest_filepath)


def load_snapshot(trainer, run_dir, ckpt=None, suffix="", load_latest=False):

    assert ckpt or load_latest

    loaded_ckpt_success = False

    if not ckpt:
        if load_latest:
            # Restore from the latest checkpoint (pointing to it by path)
            ckpt_fp = os.path.join(
                run_dir,
                "ckpts",
                "latest_checkpoint{}.pkl".format("." + suffix if suffix != "" else ""),
            )
            if os.path.isfile(ckpt_fp):
                trainer.restore(ckpt_fp)
                loaded_ckpt_success = True
                logger.info(
                    "load_snapshot -> loading %s SUCCESS for %s %s",
                    ckpt_fp,
                    suffix,
                    trainer,
                )
            else:
                logger.info(
                    "load_snapshot -> loading %s FAILED,"
                    " skipping restoring cpkt for %s %s",
                    ckpt_fp,
                    suffix,
                    trainer,
                )
        else:
            raise NotImplementedError
    elif ckpt:
        if os.path.isfile(ckpt):
            trainer.restore(ckpt)
            loaded_ckpt_success = True
            logger.info(
                "load_snapshot -> loading %s SUCCESS for %s %s", ckpt, suffix, trainer
            )
        else:
            logger.info(
                "load_snapshot -> loading %s FAILED,"
                " skipping restoring cpkt for %s %s",
                ckpt,
                suffix,
                trainer,
            )
    else:
        raise AssertionError

    # Also load snapshots of each environment object
    remote_env_fun(
        trainer,
        lambda env_wrapper: env_wrapper.load_game_object(
            os.path.join(run_dir, "ckpts")
        ),
    )

    return loaded_ckpt_success


def dump_dict(obj, run_dir, fn):

    assert isinstance(obj, dict)

    _fn = os.path.join(run_dir, fn)
    if _fn[-5:] != ".yaml":
        _fn += ".yaml"

    with open(_fn, "w") as f:
        yaml.dump(obj, f)

    print(">>> dump_dict", type(obj), run_dir, fn)


def dump_as_pkl(obj, run_dir, fn):
    _fn = os.path.join(run_dir, fn)
    if _fn[-5:] != ".pkl":
        _fn += ".pkl"
    with open(_fn, "wb") as f:
        pickle.dump(obj, f)

    print(">>> dump_as_pkl", type(obj), run_dir, fn)
