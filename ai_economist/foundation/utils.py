# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import json

import lz4.frame

from ai_economist.foundation.base.base_env import BaseEnvironment


def save_episode_log(game_object, filepath, compression_level=16):
    """Save an lz4 compressed version of the dense log stored
    in the provided game object"""
    assert isinstance(game_object, BaseEnvironment)
    compression_level = int(compression_level)
    if compression_level < 0:
        compression_level = 0
    elif compression_level > 16:
        compression_level = 16

    with lz4.frame.open(
        filepath, mode="wb", compression_level=compression_level
    ) as log_file:
        log_bytes = bytes(
            json.dumps(
                game_object.previous_episode_dense_log, ensure_ascii=False
            ).encode("utf-8")
        )
        log_file.write(log_bytes)


def load_episode_log(filepath):
    """Load the dense log saved at provided filepath"""
    with lz4.frame.open(filepath, mode="rb") as log_file:
        log_bytes = log_file.read()
    return json.loads(log_bytes)
