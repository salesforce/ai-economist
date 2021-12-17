Copyright (c) 2021, salesforce.com, inc.  
All rights reserved.  
SPDX-License-Identifier: BSD-3-Clause  
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

# Introduction
This document describes the run configuration parameters used to implement two-level curriculum learning in the [paper](https://arxiv.org/abs/2108.02755): "The AI Economist: Optimal Economic Policy Design via Two-level Deep Reinforcement Learning".

The run configurations include the `environment`, `general`, `trainer` and the agent and planner `policy`-related parameters. 
It is helpful to first go through our [tutorial](../two_level_curriculum_learning_with_rllib.md) on two-level curriculum learning to understand how we used these configurations in conjunction with with a [training script](training_script.py) to perform training. For example configurations that we used in our two-phased training approach, please see the [phase one](phase1/config.yaml) and [phase two](phase2/config.yaml) configurations.


# Descriptions of the Run Configuration Parameters

## Environment

### Base Environment
- `allow_observation_scaling` (bool): Whether to enable certain observation fields to be scaled to a range better suited for deep RL. Defaults to True.
- `components` (list): A list of tuples ("Component Name", {Component kwargs}) or
    list of dicts {"Component Name": {Component kwargs}} specifying the
    components that the instantiated environment will include.
    "Component Name" must be a string matching the name of a registered
    Component class.
    {Component kwargs} must be a dictionary of kwargs that can be passed as
    arguments to the Component class with name "Component Name".
    Resetting, stepping, and observation generation will be carried out in
    the order in which components are listed. This should be considered,
    as re-ordering the components list may impact the dynamics of the
    environment.
- `dense_log_frequency` (int): [optional] How often (in completed episodes) to
    create a dense log while playing an episode. By default, dense logging is
    turned off (dense_log_frequency=None). If dense_log_frequency=20,
    a dense log will be created when the total episode count is a multiple of
    20.
    Dense logs provide a log of agent states, actions, and rewards at each
    timestep of an episode. They also log world states at a coarser timescale
    (see below). Component classes optionally contribute additional
    information to the dense log.
    Note: dense logging is time consuming (especially with many agents).    
- `episode_length` (int): Number of timesteps in a single episode.
- `flatten_masks` (bool): Whether to flatten action masks into a single array or
    to keep as a {"action_subspace_name": action_subspace_mask} dictionary.
    For integration with deep RL, it is helpful to set this to True, for the
    purpose of action masking: flattened masks have the same semantics as
    policy logits.
- `flatten_observations` (bool): Whether to preprocess observations by
    concatenating all scalar/vector observation subfields into a single
    "flat" observation field. If not, return observations as minimally
    processed dictionaries.
- `multi_action_mode_agents` (bool): Whether mobile agents use multi_action_mode.
- `multi_action_mode_planner` (bool): Whether the planner uses multi_action_mode.
- `n_agents` (int): The number of mobile agents (does not include planner).
    Number of agents must be > 1.
- `world_dense_log_frequency` (int): When dense logging, how often (in timesteps) to log a snapshot of the world state. If world_dense_log_frequency=50 (the default), the world state will be included in the dense log for timesteps where t is a multiple of 50. Note: More frequent world snapshots increase the dense log memory footprint.
- `world_size` (list): A length-2 list specifying the dimensions of the 2D world.
    Interpreted as [height, width].

### Build Component
- `build_labor` (float): Labor cost associated with building a house.
    Must be >= 0. Default is 10.
- `payment` (int): Default amount of coin agents earn from building.
    Must be >= 0. Default is 10.
- `payment_max_skill_multiplier` (int): Maximum skill multiplier that an agent
    can sample. Must be >= 1. Default is 1.
- `skill_dist` (str): Distribution type for sampling skills. Default ("none")
    gives all agents identical skill equal to a multiplier of 1. "pareto" and
    "lognormal" sample skills from the associated distributions.

### ContinuousDoubleAuction Component
- `max_bid_ask` (int): Maximum amount of coin that an agent can bid or ask for.
    Must be >= 1. Default is 10 coin.
- `max_num_orders` (int, optional): Maximum number of bids + asks that an agent can have open for a given resource. Must be >= 1. Default is no limit to number of orders.
- `order_labor` (float): Amount of labor incurred when an agent creates an order.
    Must be >= 0. Default is 0.25.
- `order_duration` (int): Number of environment timesteps before an unfilled
    bid/ask expires. Must be >= 1. Default is 50 timesteps.

### Gather Component
- `collect_labor` (float): Labor cost associated with collecting resources. This
    cost is added (in addition to any movement cost) when the agent lands on
    a tile that is populated with resources (triggering collection).
    Must be >= 0. Default is 1.0.
- `move_labor` (float): Labor cost associated with movement. Must be >= 0.
    Default is 1.0.
- `skill_dist` (str): Distribution type for sampling skills. Default ("none")
    gives all agents identical skill equal to a bonus prob of 0. "pareto" and
    "lognormal" sample skills from the associated distributions.

### PeriodicBracketTax Component
- `bracket_spacing` (str): How bracket cutoffs should be spaced.
    "us-federal" (default) uses scaled cutoffs from the 2018 US Federal
        taxes, with scaling set by usd_scaling (ignores n_brackets and
        top_bracket_cutoff);
    "linear" linearly spaces the n_bracket cutoffs between 0 and
        top_bracket_cutoff;
    "log" is similar to "linear" but with logarithmic spacing.
- `disable_taxes` (bool): Whether to disable any tax collection, effectively
    enforcing that tax rates are always 0. Useful for removing taxes without
    changing the observation space. Default is False (taxes enabled).
- `fixed_bracket_rates` (list): Required if tax_model=="fixed-bracket-rates". A
    list of fixed marginal rates to use for each bracket. Length must be
    equal to the number of brackets (7 for "us-federal" spacing, n_brackets
    otherwise).
- `n_brackets` (int): How many tax brackets to use. Must be >=2. Default is 5.
- `pareto_weight_type` (str): Type of pareto weights to use when computing tax
    rates using the Saez formula. "inverse_income" (default) uses 1/z;
    "uniform" uses 1.
- `period` (int): Length of a tax period in environment timesteps. Taxes are
    updated at the start of each period and collected/redistributed at the
    end of each period. Must be > 0. Default is 100 timesteps.
- `rate_disc` (float): (Only applies for "model_wrapper") the interval separating
    discrete tax rates that the planner can select. Default of 0.05 means,
    for example, the planner can select among [0.0, 0.05, 0.10, ... 1.0].
    Must be > 0 and < 1.
- `rate_min` (float): Minimum tax rate within a bracket. Must be >= 0 (default).
- `rate_max` (float): Maximum tax rate within a bracket. Must be <= 1 (default).
- `saez_fixed_elas` (float, optional): If supplied, this value will be used as
    the elasticity estimate when computing tax rates using the Saez formula.
    If not given (default), elasticity will be estimated empirically.
- `tax_annealing_schedule` (list, optional): A length-2 list of
    [tax_annealing_warmup, tax_annealing_slope] describing the tax annealing
    schedule. See annealed_tax_mask function for details. Default behavior is
    no tax annealing.
- `tax_model` (str): Which tax model to use for setting taxes.
    "model_wrapper" (default) uses the actions of the planner agent;
    "saez" uses an adaptation of the theoretical optimal taxation formula
    derived in this [paper](https://www.nber.org/papers/w7628).
    "us-federal-single-filer-2018-scaled" uses US federal tax rates from 2018;
    "fixed-bracket-rates" uses the rates supplied in fixed_bracket_rates.
- `top_bracket_cutoff` (float): The income at the left end of the last tax
    bracket. Must be >= 10. Default is 100 coin.
- `usd_scaling` (float): Scale by which to divide the US Federal bracket cutoffs
    when using bracket_spacing = "us-federal". Must be > 0. Default is 1000.

### Gather-Trade-Build Scenario
- `energy_cost` (float): Coefficient for converting labor to negative utility.
- `energy_warmup_constant` (float): Decay constant that controls the rate at which the effective energy cost is annealed from 0 to energy_cost. Set to 0 (default) to disable annealing, meaning that the effective energy cost is always energy_cost. The units of the decay constant depend on the choice of energy_warmup_method.
- `energy_warmup_method` (str): How to schedule energy annealing (warmup). If
    "decay" (default), use the number of completed episodes. If "auto",
    use the number of timesteps where the average agent reward was positive.
- `env_layout_file` (str): Name of the layout file in ./map_txt/ to use.
    Note: The world dimensions of that layout must match the world dimensions
    argument used to construct the environment.
- `fixed_four_skill_and_loc` (bool): Whether to use a fixed set of build skills and
    starting locations with 4 agents. False, by default.
    Note: Requires that n_agents=4 and that the environment uses the "Build"
    component with skill_dist="pareto".
- `full_observability` (bool): Whether the mobile agents' spatial observation
    includes the full world view or is instead an egocentric view.
- `isoelastic_eta` (float): Parameter controlling the shape of agent utility
    wrt coin endowment.
- `mixing_weight_gini_vs_coin` (float): Degree to which equality is ignored w/
    "coin_eq_times_productivity". Default is 0, which weights equality and
    productivity equally. If set to 1, only productivity is rewarded.
- `mobile_agent_observation_range` (int): If not using full_observability,
    the spatial range (on each side of the agent) that is visible in the
    spatial observations.
- `planner_gets_spatial_obs` (bool): Whether the planner agent receives spatial
    observations from the world.
- `planner_reward_type` (str): The type of reward used for the planner. Options
    are "coin_eq_times_productivity" (default),
    "inv_income_weighted_coin_endowment", and "inv_income_weighted_utility".
- `resource_regen_prob` (float): Probability that an empty source tile will
    regenerate a new resource unit.
- `starting_agent_coin` (int, float): Amount of coin agents have at t=0. Defaults
    to zero coin.

## General
- `ckpt_frequency_steps` (int): Specify how frequently (in environment steps) to save the trained model checkpoints.
- `cpus` (int): Number of  CPUs in the system.
- `episodes` (int): Number of episodes to run the training for.
- `gpus` (int): Number of GPUs in the system.
- `restore_tf_weights_agents` (filepath): Path to agent model checkpoint (saved via TensorFlow (TF)). When specified, training resumes after restoring the agent (TF) weights, otherwise it starts with fresh agent weights.
- `restore_tf_weights_planner` (filepath): Path to planner model checkpoint (saved via TensorFlow (TF)). When specified, training resumes after restoring the planner (TF) weights, otherwise it starts with fresh agent weights.
- `train_planner` (bool): Flag to specify whether to train only the agents (when False) or train both the agents and the planner (when True).

## Trainer
- `batch_mode` (str):  Whether to rollout "complete_episodes" or "truncate_episodes" to  "rollout_fragment_length" length unrolls. Episode truncation guarantees evenly sized batches, but increases variance as the reward-to-go will need to be estimated at truncation boundaries.
- `env_config` (dict): Arguments to pass to the env creator. <br>
Note: this is updated in the [training script](training_script.py).
- `local_tf_session_args`: Configures TF for single-process operation by default. <br>
Note: These are settings related to TF and are documented [here](https://github.com/tensorflow/tensorflow/blob/26b4dfa65d360f2793ad75083c797d57f8661b93/tensorflow/core/protobuf/config.proto#L165).
    - `inter_op_parallelism_threads` (int)
    - `intra_op_parallelism_threads` (int) <br>
- `metrics_smoothing_episodes` (int): Smooth metrics over this many episodes. <br>
Note: this is updated in the [training script](training_script.py).
- `multiagent` (dict): Settings for Multi-Agent Environments. This is a dictionary containing <br>
    a) Map from policy ids to tuples of (policy_cls, obs_space, act_space, 
    config). <br>
    b) Function mapping agent ids to policy ids. <br>
    c) Optional whitelist of policies to train, or None for all policies. <br>
    Note: this is updated in the [training script](training_script.py).
- `no_done_at_end` (bool): Don't set 'done' at the end of the episode. Note that you still need to set this if "soft_horizon=True", unless your env is actually running forever without returning "done=True".
- `num_envs_per_worker` (int): Number of environments to evaluate vectorwise per worker. This enables model inference batching, which can improve performance for inference bottlenecked workloads.
- `num_gpus` (int):  Number of GPUs to allocate to the trainer process. Note that not all algorithms can take advantage of trainer GPUs. This can be fractional (e.g., 0.3 GPUs).
- `num_gpus_per_worker` (int): Number of GPUs to allocate per worker. This can be fractional. This is usually needed only if your env itself requires a GPU (i.e., it is a GPU-intensive video game), or model inference is unusually expensive.
- `num_sgd_iter` (int): Number of SGD iterations in each outer loop.
 - `num_workers` (int): Number of rollout worker actors to create for parallel sampling. Setting this to 0 will force rollouts to be done in the trainer actor.
 - `observation_filter` (str): Element-wise observation filter, either "NoFilter" or "MeanStdFilter".
 - `rollout_fragment_length` (int): Divide episodes into fragments of this many steps each during rollouts. Sample batches of this size are collected from rollout workers and  combined into a larger batch of "train_batch_size" for learning.
    For example, given rollout_fragment_length=100 and train_batch_size=1000:
      1. RLlib collects 10 fragments of 100 steps each from rollout workers.
      2. These fragments are concatenated and we perform an epoch of SGD.
    When using multiple envs per worker, the fragment size is multiplied by
    "num_envs_per_worker". This is since we are collecting steps from
    multiple envs in parallel. For example, if num_envs_per_worker=5, then
    rollout workers will return experiences in chunks of 5*100 = 500 steps.
    The dataflow here can vary per algorithm. For example, PPO further
    divides the train batch into minibatches for multi-epoch SGD.
- `seed` (int): This argument, in conjunction with worker_index, sets the random seed of  each worker, so that identically configured trials will have identical  results. This makes experiments reproducible.
- `sgd_minibatch_size` (int): Total SGD batch size across all devices for SGD.
- `shuffle_sequences` (bool): Whether to shuffle sequences in the batch when training. It is recommended to set this to True.
- `tf_session_args`: Configures TF for single-process operation by default. Note: This is overwritten by `local_tf_session_args`. These are settings related to TF and are documented [here](https://github.com/tensorflow/tensorflow/blob/26b4dfa65d360f2793ad75083c797d57f8661b93/tensorflow/core/protobuf/config.proto#L165).
    - `allow_soft_placement` (bool)
    - `device_count`:
        - `CPU` (int)
        - `GPU` (int)
    - `gpu_options`:
        - `allow_growth` (bool)
    - `inter_op_parallelism_threads` (int)
    - `intra_op_parallelism_threads` (int)
    - `log_device_placement` (bool)
- `train_batch_size` (int): Training batch size, if applicable. Should be >= rollout_fragment_length. Samples batches will be concatenated together to a batch of this size, which is then passed to SGD.

## Agent and Planner Policy
- `clip_param` (float): PPO clip parameter.
- `entropy_coeff` (float): Coefficient of the entropy loss.
- `entropy_coeff_schedule` (list of tuples): Entropy schedule.
- `gamma` (float): Discount factor of the MDP.
- `grad_clip` (float): Clip the global norm of gradients by this amount.
- `kl_coeff` (float): Initial coefficient for KL divergence.
- `kl_target` (float): Target value for KL divergence.
- `lambda` (float): GAE(lambda) parameter.
- `lr` (float): Stepsize of SGD.
- `lr_schedule` (list of tuples): Learning rate schedule.
- `model`:
    - `custom_model` (str): Registered policy model name. Current options are "keras_conv_lstm" or "random".
    - `custom_options`:
      - `fc_dim` (int): Dimension of the fully-connected layer.
      - `idx_emb_dim` (int): Output embedding dimension.
      - `input_emb_vocab` (int): Input embedding dimension.
      - `lstm_cell_size` (int): Size of the LSTM cell.
      - `num_conv` (int): Number of convolutional layers.
      - `num_fc` (int): Number of fully-connected layers.
    - `max_seq_len` (int): Maximum seq len for training the LSTM.
- `use_gae` (bool): If true, use the Generalized Advantage Estimator ([GAE](https://arxiv.org/abs/1506.02438)) with a value function.
- `vf_clip_param` (float): Clip param for the value function. Note that this is sensitive to the scale of the rewards. If your expected value function is large, increase this.
- `vf_loss_coeff` (float): Coefficient of the value function loss. Important: you must tune this if you set "vf_share_layers: True".
- `vf_share_layers` (bool): Share layers for value function.

## Additional Configurations
(For additional configurations, see RLlib [common parameters](https://docs.ray.io/en/releases-0.8.4/rllib-training.html#common-parameters), [PPO-specific configs](https://docs.ray.io/en/releases-0.8.4/rllib-algorithms.html#ppo) and [model parameters](https://docs.ray.io/en/releases-0.8.4/rllib-models.html?highlight=custom%20models#built-in-model-parameters). 