# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import itertools
import os
import random
from pathlib import Path

import numpy as np
import pycuda
import pycuda.autoinit
import pycuda.driver as cuda_driver
import scipy
import scipy.stats
import torch
from pycuda.compiler import SourceModule
from torch.distributions import Categorical
from tqdm import tqdm

from .constants import (
    consumer_state_scaling_factors,
    firm_state_scaling_factors,
    govt_state_scaling_factors,
)
from .networks import DeterministicPolicy, IndependentPolicyNet, PolicyNet
from .util import expand_to_digit_form, size_after_digit_expansion

_NP_DTYPE = np.float32

# the below line is 'strangely' necessary to make PyTorch work with PyCUDA
pytorch_cuda_init_success = torch.cuda.FloatTensor(8)


# for opening source files within module
module_path = Path(__file__).parent


def interval_list_contains(interval_list, step):
    for (lower, upper_non_inclusive) in interval_list:
        if lower <= step < upper_non_inclusive:
            return True
    return False


class NoOpOptimizer:
    """
    Dummy Optimizer.
    """

    def __init__(self):
        pass

    def step(self):
        pass


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def reverse_cumsum(x):
    # assumes summing along episode iteration dim
    return x + torch.sum(x, dim=-2, keepdims=True) - torch.cumsum(x, dim=-2)


def discounted_returns(rewards, gamma):
    maxt = rewards.shape[-2]
    cumulative_rewards = 0
    returns = torch.zeros_like(rewards)
    for t in reversed(range(maxt)):
        returns[:, t, :] = rewards[:, t, :] + gamma * cumulative_rewards
        cumulative_rewards = rewards[:, t, :] + cumulative_rewards
    return returns


def compute_theta_coef(hparams_dict, episode):
    anneal_dict = hparams_dict["agents"]["consumer_anneal_theta"]
    if anneal_dict["anneal_on"]:
        exp_decay_length_in_steps = anneal_dict["exp_decay_length_in_steps"]
        theta_coef = np.float32(1.0 - (np.exp(-episode / exp_decay_length_in_steps)))
    else:
        return np.float32(1.0)
    return theta_coef


def government_action_mask(hparams_dict, step):
    government_actions_array = hparams_dict["agents"]["government_actions_array"]
    tax_annealing_params = hparams_dict["agents"]["government_anneal_taxes"]

    income_tax = torch.tensor(government_actions_array[:, 0]).cuda()
    corporate_tax = torch.tensor(government_actions_array[:, 1]).cuda()
    mask = torch.zeros(income_tax.shape[0]).cuda()

    if not tax_annealing_params["anneal_on"]:
        return None
    a0 = tax_annealing_params["start"]
    max_tax = tax_annealing_params["increase_const"] * step + a0
    mask[(income_tax > max_tax) | (corporate_tax > max_tax)] -= 1000.0

    return mask


def firm_action_mask(hparams_dict, step):
    # pick out all firm actions where wage is the wrong height,
    # and assign -1000.0 to those
    firm_actions_array = hparams_dict["agents"]["firm_actions_array"]
    wage_annealing_params = hparams_dict["agents"]["firm_anneal_wages"]
    price_annealing_params = hparams_dict["agents"]["firm_anneal_prices"]
    wages = torch.tensor(firm_actions_array[:, 1]).cuda()
    prices = torch.tensor(firm_actions_array[:, 0]).cuda()
    mask = torch.zeros(wages.shape[0]).cuda()

    if not (wage_annealing_params["anneal_on"] or price_annealing_params["anneal_on"]):
        return None

    if wage_annealing_params["anneal_on"]:
        a0 = wage_annealing_params["start"]
        max_wage = wage_annealing_params["increase_const"] * step + a0
        min_wage = -wage_annealing_params["decrease_const"] * step + a0
        mask[(wages < min_wage) | (wages > max_wage)] -= 1000.0
    if price_annealing_params["anneal_on"]:
        a0 = price_annealing_params["start"]
        max_price = price_annealing_params["increase_const"] * step + a0
        min_price = -price_annealing_params["decrease_const"] * step + a0
        mask[(prices < min_price) | (prices > max_price)] -= 1000.0

    return mask


def get_cuda_code(rel_path_to_cu_file, **preprocessor_vars_to_replace):
    with open(module_path / rel_path_to_cu_file) as cudasource:
        code_string = cudasource.read()

    # format for preprocessor macros in firm_rbc.cu is M_VARNAME.
    # Specify all these as args to nvcc.
    options_list = [
        f"-D M_{k.upper()}={v}" for k, v in preprocessor_vars_to_replace.items()
    ]

    return code_string, options_list


def add_penalty_for_no_ponzi(
    states, rewards, budget_offset, penalty_coef=20.0, penalty_scale=100.0
):
    budget_violations = -torch.clamp_max(states[..., budget_offset], 0.0)
    rewards[:, -1, :] -= penalty_coef * budget_violations / penalty_scale


def update_government_rewards(
    government_rewards, consumer_rewards, firm_rewards, cfg_dict
):
    assert (
        government_rewards == 0.0
    ).all()  # govt should have been assigned exactly 0 in cuda step function
    total_rewards = consumer_rewards.sum(dim=-1)
    if cfg_dict["agents"]["government_counts_firm_reward"] == 1:
        total_rewards = total_rewards + cfg_dict["agents"].get(
            "firm_reward_for_government_factor", 1.0
        ) * firm_rewards.sum(dim=-1)

    government_rewards[..., 0] = total_rewards[:]  # one govt for now


def update_penalty_coef(
    states,
    budget_offset,
    prev_penalty_coef,
    penalty_step_size=0.01,
    penalty_scale=100.0,
):
    budget_violations = -torch.clamp_max(states[..., budget_offset], 0.0)
    new_penalty_coef = (
        prev_penalty_coef
        + penalty_step_size * (budget_violations / penalty_scale).mean().item()
    )
    return new_penalty_coef


def get_actions_from_inds(action_inds, agents_dict):

    _action_inds = action_inds.cpu().to(torch.long)

    consumption_action_tensor = torch.tensor(
        agents_dict["consumer_consumption_actions_array"]
    )

    work_action_tensor = torch.tensor(agents_dict["consumer_work_actions_array"])
    num_firms = agents_dict["num_firms"]
    out_shape = _action_inds.shape[:-1] + (agents_dict["consumer_action_dim"],)
    consumer_actions_out = torch.zeros(out_shape)
    idx_hours_worked = num_firms
    idx_which_firm = num_firms + 1

    for i in range(num_firms):
        consumer_actions_out[..., i] = consumption_action_tensor[
            _action_inds[..., i], :
        ].squeeze(dim=-1)

    consumer_actions_out[..., num_firms] = work_action_tensor[
        _action_inds[..., idx_hours_worked], :
    ].squeeze(dim=-1)

    consumer_actions_out[..., (num_firms + 1)] = _action_inds[..., idx_which_firm]

    return consumer_actions_out


def anneal_entropy_coef(entropy_dict, step):
    if entropy_dict is None:
        return 1.0

    if entropy_dict["anneal_on"]:
        coef_floor = entropy_dict.get("coef_floor", 0.0)
        return max(
            np.exp(-step / entropy_dict["exp_decay_length_in_steps"]), coef_floor
        )
    return 1.0


def get_grad_norm(policy):
    grad_norm = 0.0
    for p in list(filter(lambda p: p.grad is not None, policy.parameters())):
        grad_norm += (p.grad.data.norm(2).item()) ** 2
    return grad_norm


def get_ev(adv, returns, cutoff=-1.0):
    return max(cutoff, (1 - (adv.detach().var() / returns.detach().var())).item())


def consumer_ppo_step(
    policy,
    states,
    actions,
    rewards,
    optimizer,
    gamma_const,
    entropy_val=0.0,
    value_loss_weight=1.0,
    ppo_num_updates=3,
    reward_scale=1.0,
    clip_grad_norm=None,
    clip_param=0.1,
):
    # Get initial policy predictions
    multi_action_probs, old_value_preds = policy(states)

    old_value_preds = old_value_preds.detach()
    # Get returns
    rescaled_rewards = rewards / reward_scale
    G_discounted_returns = discounted_returns(rescaled_rewards, gamma_const)

    # Value function loss

    sum_old_log_probs = 0.0
    for action_ind, probs in enumerate(multi_action_probs):
        _CategoricalDist = Categorical(probs)
        sum_old_log_probs += -1.0 * _CategoricalDist.log_prob(actions[..., action_ind])
    sum_old_log_probs = sum_old_log_probs.detach()

    assert not G_discounted_returns.requires_grad
    assert not sum_old_log_probs.requires_grad
    assert not old_value_preds.requires_grad

    # Compute ppo loss
    for _ in range(ppo_num_updates):
        multi_action_probs, value_preds = policy(states)
        get_huber_loss = torch.nn.SmoothL1Loss()
        value_pred_clipped = old_value_preds + (value_preds - old_value_preds).clamp(
            -clip_param, clip_param
        )
        value_loss_new = get_huber_loss(
            value_preds.squeeze(dim=-1), G_discounted_returns
        )  # can use huber loss instead
        value_loss_clipped = get_huber_loss(
            value_pred_clipped.squeeze(dim=-1), G_discounted_returns
        )

        value_loss = torch.max(value_loss_new, value_loss_clipped).mean()

        # Policy loss with value function baseline.
        advantages = G_discounted_returns - value_preds.detach().squeeze(dim=-1)
        # Don't propagate through to VF network.
        assert not advantages.requires_grad

        # Trick: standardize advantages
        standardized_advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-6
        )
        sum_mean_entropy = 0.0  # mean over batch and agents
        sum_neg_log_probs = 0.0

        for action_ind, probs in enumerate(multi_action_probs):
            _CategoricalDist = Categorical(probs)
            sum_neg_log_probs += -1.0 * _CategoricalDist.log_prob(
                actions[..., action_ind]
            )
            sum_mean_entropy += _CategoricalDist.entropy().mean()

        assert sum_neg_log_probs.requires_grad
        # note: log probs are negative, so negate again here
        ratio = torch.exp(-sum_neg_log_probs + sum_old_log_probs)
        surr1 = ratio * standardized_advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
            * standardized_advantages
        )

        ppo_loss = -torch.min(surr1, surr2).mean()

        loss = (
            ppo_loss - entropy_val * sum_mean_entropy + value_loss_weight * value_loss
        )

        # Apply gradients
        optimizer.zero_grad()
        loss.backward()

        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=clip_grad_norm)

        optimizer.step()


def ppo_step(
    policy,
    states,
    actions,
    rewards,
    optimizer,
    gamma_const,
    entropy_val=0.0,
    value_loss_weight=1.0,
    ppo_num_updates=3,
    actions_mask=None,
    reward_scale=1.0,
    clip_grad_norm=None,
    clip_param=0.1,
):
    # Get initial policy predictions
    probs, old_value_preds = policy(states, actions_mask=actions_mask)
    old_value_preds = old_value_preds.detach()

    # Get returns
    rescaled_rewards = rewards / reward_scale
    G_discounted_returns = discounted_returns(rescaled_rewards, gamma_const)

    # Value function loss

    _CategoricalDist = Categorical(probs)
    old_log_probs = -1.0 * _CategoricalDist.log_prob(actions).detach()

    assert not G_discounted_returns.requires_grad
    assert not old_log_probs.requires_grad
    assert not old_value_preds.requires_grad

    # Compute ppo loss
    for _ in range(ppo_num_updates):
        probs, value_preds = policy(states, actions_mask=actions_mask)
        get_huber_loss = torch.nn.SmoothL1Loss()
        value_pred_clipped = old_value_preds + (value_preds - old_value_preds).clamp(
            -clip_param, clip_param
        )
        value_loss_new = get_huber_loss(
            value_preds.squeeze(dim=-1), G_discounted_returns
        )  # can use huber loss instead
        value_loss_clipped = get_huber_loss(
            value_pred_clipped.squeeze(dim=-1), G_discounted_returns
        )

        value_loss = torch.max(value_loss_new, value_loss_clipped).mean()

        # Policy loss with value function baseline.
        advantages = G_discounted_returns - value_preds.detach().squeeze(dim=-1)
        # Don't propagate through to VF network.
        assert not advantages.requires_grad

        # Trick: standardize advantages
        standardized_advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-6
        )

        _CategoricalDist = Categorical(probs)
        neg_log_probs = -1.0 * _CategoricalDist.log_prob(actions)
        mean_entropy = _CategoricalDist.entropy().mean()

        assert neg_log_probs.requires_grad
        # note: log probs are negative, so negate again here
        ratio = torch.exp(-neg_log_probs + old_log_probs)
        surr1 = ratio * standardized_advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
            * standardized_advantages
        )

        ppo_loss = -torch.min(surr1, surr2).mean()

        loss = ppo_loss - entropy_val * mean_entropy + value_loss_weight * value_loss

        # Apply gradients
        optimizer.zero_grad()
        loss.backward()

        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=clip_grad_norm)

        optimizer.step()


def consumer_policy_gradient_step(
    policy,
    states,
    actions,
    rewards,
    optimizer,
    gamma_const,
    entropy_val=0.0,
    value_loss_weight=1.0,
    reward_scale=1.0,
    clip_grad_norm=None,
):
    # Get policy and value predictions
    multi_action_probs, value_preds = policy(states)

    # Get returns
    rescaled_rewards = rewards / reward_scale
    G_discounted_returns = discounted_returns(rescaled_rewards, gamma_const)

    # Value function loss
    get_huber_loss = torch.nn.SmoothL1Loss()
    value_loss = get_huber_loss(
        value_preds.squeeze(dim=-1), G_discounted_returns
    ).mean()  # can use huber loss instead

    # Policy loss with value function baseline.
    advantages = G_discounted_returns - value_preds.detach().squeeze(dim=-1)
    # Don't propagate through to VF network.
    assert not advantages.requires_grad

    # Trick: standardize advantages
    standardized_advantages = (advantages - advantages.mean()) / (
        advantages.std() + 1e-6
    )

    # Compute policy loss
    sum_mean_entropy = 0.0  # mean over batch and agents
    sum_neg_log_probs = 0.0

    for action_ind, probs in enumerate(multi_action_probs):
        _CategoricalDist = Categorical(probs)
        sum_neg_log_probs += -1.0 * _CategoricalDist.log_prob(actions[..., action_ind])
        sum_mean_entropy += _CategoricalDist.entropy().mean()

    pg_loss = (sum_neg_log_probs * standardized_advantages).mean()
    assert sum_neg_log_probs.requires_grad

    loss = pg_loss - entropy_val * sum_mean_entropy + value_loss_weight * value_loss

    # Apply gradients
    optimizer.zero_grad()
    loss.backward()

    if clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=clip_grad_norm)

    optimizer.step()


def policy_gradient_step(
    policy,
    states,
    actions,
    rewards,
    optimizer,
    gamma_const,
    entropy_val=0.0,
    value_loss_weight=1.0,
    actions_mask=None,
    reward_scale=1.0,
    clip_grad_norm=None,
):

    # here, we must perform digit scaling
    optimizer.zero_grad()
    probs, value_preds = policy(states, actions_mask=actions_mask)
    rewards = rewards / reward_scale
    G_discounted_returns = discounted_returns(rewards, gamma_const)
    get_huber_loss = torch.nn.SmoothL1Loss()
    value_loss = get_huber_loss(
        value_preds.squeeze(dim=-1), G_discounted_returns
    ).mean()  # can use huber loss instead
    advantages = G_discounted_returns - value_preds.detach().squeeze(
        dim=-1
    )  # compute advantages (don't propagate through to VF network)
    assert not advantages.requires_grad
    # mean and standardize advantages
    standardized_advantages = (advantages - advantages.mean()) / (
        advantages.std() + 1e-6
    )
    assert not standardized_advantages.requires_grad
    m = Categorical(probs)
    pg_loss = (-m.log_prob(actions) * standardized_advantages).mean()
    assert pg_loss.requires_grad
    entropy_regularize = entropy_val * m.entropy().mean()
    loss = pg_loss - entropy_regularize + value_loss_weight * value_loss
    loss.backward()

    if clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=clip_grad_norm)

    optimizer.step()


def save_dense_log(
    save_dir,
    epi,
    agent_type_arrays,
    agent_action_arrays,
    agent_aux_arrays,
):
    print(f"Saving dense log at episode {epi}")
    for agent_type in ["consumer", "firm", "government"]:
        states_batch, actions_batch, rewards_batch = agent_type_arrays[agent_type]
        aux_array = agent_aux_arrays[agent_type]
        if aux_array is not None:
            aux_array = aux_array.cpu().numpy()
        np.savez(
            str(Path(save_dir) / Path(f"episode_{epi}_{agent_type}.npz")),
            states=states_batch.cpu().numpy(),
            actions=actions_batch.cpu().numpy(),
            rewards=rewards_batch.cpu().numpy(),
            action_array=agent_action_arrays[agent_type],
            aux_array=aux_array,
        )


def save_policy_parameters(
    save_dir,
    epi,
    consumer_policy,
    firm_policy,
    government_policy,
    freeze_firms,
    freeze_govt,
):
    print(f"saving model parameters at episode {epi}")
    consumer_path = (
        Path(save_dir) / Path("saved_models") / Path(f"consumer_policy_{epi}.pt")
    )

    # always save the latest, to be overwritten later
    consumer_path_latest = (
        Path(save_dir) / Path("saved_models") / Path("consumer_policy_latest.pt")
    )
    os.makedirs(consumer_path.parent, exist_ok=True)
    torch.save(consumer_policy.state_dict(), consumer_path)
    torch.save(consumer_policy.state_dict(), consumer_path_latest)

    if freeze_firms is None:
        firm_path = (
            Path(save_dir) / Path("saved_models") / Path(f"firm_policy_{epi}.pt")
        )
        firm_path_latest = (
            Path(save_dir) / Path("saved_models") / Path("firm_policy_latest.pt")
        )

        os.makedirs(firm_path.parent, exist_ok=True)
        torch.save(firm_policy.state_dict(), firm_path)
        torch.save(firm_policy.state_dict(), firm_path_latest)
    if freeze_govt is None:
        government_path = (
            Path(save_dir) / Path("saved_models") / Path(f"government_policy_{epi}.pt")
        )
        government_path_latest = (
            Path(save_dir) / Path("saved_models") / Path("government_policy_latest.pt")
        )

        os.makedirs(government_path.parent, exist_ok=True)
        torch.save(government_policy.state_dict(), government_path)
        torch.save(government_policy.state_dict(), government_path_latest)


class ConsumerFirmRunManagerBatchParallel:
    """
    The Real Business Cycle Experiment Management Class.
    """

    def __init__(self, cfg_dict, freeze_firms=None, freeze_govt=None):
        self.cfg_dict = cfg_dict
        self.train_dict = cfg_dict["train"]
        self.agents_dict = cfg_dict["agents"]
        self.world_dict = cfg_dict["world"]
        self.save_dense_every = self.train_dict["save_dense_every"]
        self.save_dir = self.train_dict["save_dir"]

        self.freeze_firms = freeze_firms
        self.freeze_govt = freeze_govt

        self.__init_cuda_functions()
        self.__init_cuda_data_structs()
        self.__init_torch_data()

    def __init_cuda_data_structs(self):
        __td = self.train_dict
        __ad = self.agents_dict
        __wd = self.world_dict
        batch_size = __td["batch_size"]
        num_consumers = __ad["num_consumers"]
        num_firms = __ad["num_firms"]
        num_governments = __ad["num_governments"]
        firm_action_dim = __ad["firm_action_dim"]
        government_action_dim = __ad["government_action_dim"]
        consumer_state_dim = __ad["consumer_state_dim"]
        firm_state_dim = __ad["firm_state_dim"]
        government_state_dim = __ad["government_state_dim"]
        global_state_dim = __ad["global_state_dim"]
        consumer_endowment = __wd["initial_consumer_endowment"]
        firm_endowment = __wd["initial_firm_endowment"]
        initial_stocks = __wd["initial_stocks"]
        initial_wages = __wd["initial_wages"]
        initial_prices = __wd["initial_prices"]
        consumer_theta = __wd["consumer_theta"]

        consumer_rewards = np.zeros((batch_size, num_consumers), dtype=_NP_DTYPE)
        consumer_states = np.zeros(
            (batch_size, num_consumers, consumer_state_dim), dtype=_NP_DTYPE
        )

        firm_action_indices = np.zeros((batch_size, num_firms), dtype=np.int32)
        firm_actions = np.zeros(
            (batch_size, num_firms, firm_action_dim), dtype=_NP_DTYPE
        )
        firm_rewards = np.zeros((batch_size, num_firms), dtype=_NP_DTYPE)
        firm_states = np.zeros((batch_size, num_firms, firm_state_dim), dtype=_NP_DTYPE)

        government_action_indices = np.zeros(
            (batch_size, num_governments), dtype=np.int32
        )
        government_actions = np.zeros(
            (batch_size, num_governments, government_action_dim), dtype=_NP_DTYPE
        )
        government_rewards = np.zeros((batch_size, num_governments), dtype=_NP_DTYPE)
        government_states = np.zeros(
            (batch_size, num_governments, government_state_dim), dtype=_NP_DTYPE
        )

        # initialize states to right values here

        # global state init
        # for consumers, firms, and governments
        for state_arr in [consumer_states, firm_states, government_states]:
            # set prices to 1.0
            state_arr[:, :, 0:num_firms] = initial_prices
            # set wages to 0.0
            state_arr[:, :, num_firms : (2 * num_firms)] = initial_wages
            # set stocks to 0.0
            state_arr[:, :, (2 * num_firms) : (3 * num_firms)] = initial_stocks
            # set goods overdemanded to 0.0
            state_arr[:, :, (3 * num_firms) : (4 * num_firms)] = 0.0
            # set taxes to 0.0
            state_arr[:, :, (4 * num_firms)] = 0.0
            state_arr[:, :, (4 * num_firms) + 1] = 0.0

        # consumer states, set theta and initial budget
        if "paretoscaletheta" in __wd:
            pareto_vals = np.expand_dims(
                scipy.stats.pareto.ppf(
                    (np.arange(num_consumers) / num_consumers), __wd["paretoscaletheta"]
                ),
                axis=0,
            )
            consumer_states[:, :, consumer_state_dim - 1] = consumer_theta * (
                1.0 / pareto_vals
            )
        else:
            consumer_states[:, :, consumer_state_dim - 1] = consumer_theta
        consumer_states[:, :, global_state_dim] = consumer_endowment

        # firm states
        # capital
        if __wd.get("initial_capital", None) == "proportional":
            for i in range(num_firms):
                firm_states[:, i, global_state_dim + 1] = ((i + 1) / 10.0) * 2.0
        elif __wd.get("initial_capital", None) == "twolevel":
            for i in range(num_firms):
                if i < (num_firms // 2):
                    firm_states[:, i, global_state_dim + 1] = 5000
                else:
                    firm_states[:, i, global_state_dim + 1] = 10000
        else:
            firm_states[:, :, global_state_dim + 1] = 1.0

        # production alpha
        if __wd["production_alpha"] == "proportional":
            half_firms = num_firms // 2
            for i in range(num_firms):
                firm_states[:, i, global_state_dim + 2] = ((i % half_firms) + 1) * 0.2
        elif __wd["production_alpha"] == "fixed_array":
            alpha_arr = [0.2, 0.3, 0.4, 0.6, 0.8, 0.2, 0.3, 0.4, 0.6, 0.8]
            for i in range(num_firms):
                firm_states[:, i, global_state_dim + 2] = alpha_arr[i]
        else:
            for i in range(num_firms):
                firm_states[:, i, global_state_dim + 2] = __wd["production_alpha"]

        # set one-hot fields correctly by index for each firm
        onehot_rows = np.eye(num_firms)
        firm_states[:, :, (global_state_dim + 3) :] = onehot_rows
        firm_states[:, :, global_state_dim] = firm_endowment

        # government states
        # for now, nothing beyond global state

        self.consumer_states_gpu_tensor = torch.from_numpy(consumer_states).cuda()
        # these are now tensors bc sampling for consumers via pytorch
        self.consumer_rewards_gpu_pycuda = cuda_driver.mem_alloc(
            consumer_rewards.nbytes
        )
        self.consumer_states_checkpoint_gpu_pycuda = cuda_driver.mem_alloc(
            consumer_states.nbytes
        )
        cuda_driver.memcpy_htod(self.consumer_rewards_gpu_pycuda, consumer_rewards)
        cuda_driver.memcpy_htod(
            self.consumer_states_checkpoint_gpu_pycuda, consumer_states
        )

        self.firm_states_gpu_tensor = torch.from_numpy(firm_states).cuda()
        self.firm_action_indices_gpu_pycuda = cuda_driver.mem_alloc(
            firm_action_indices.nbytes
        )
        self.firm_actions_gpu_pycuda = cuda_driver.mem_alloc(firm_actions.nbytes)
        self.firm_rewards_gpu_pycuda = cuda_driver.mem_alloc(firm_rewards.nbytes)
        self.firm_states_checkpoint_gpu_pycuda = cuda_driver.mem_alloc(
            firm_states.nbytes
        )
        cuda_driver.memcpy_htod(
            self.firm_action_indices_gpu_pycuda, firm_action_indices
        )
        cuda_driver.memcpy_htod(self.firm_actions_gpu_pycuda, firm_actions)
        cuda_driver.memcpy_htod(self.firm_rewards_gpu_pycuda, firm_rewards)
        cuda_driver.memcpy_htod(self.firm_states_checkpoint_gpu_pycuda, firm_states)

        self.government_states_gpu_tensor = torch.from_numpy(government_states).cuda()
        self.government_action_indices_gpu_pycuda = cuda_driver.mem_alloc(
            government_action_indices.nbytes
        )
        self.government_actions_gpu_pycuda = cuda_driver.mem_alloc(
            government_actions.nbytes
        )
        self.government_rewards_gpu_pycuda = cuda_driver.mem_alloc(
            government_rewards.nbytes
        )
        self.government_states_checkpoint_gpu_pycuda = cuda_driver.mem_alloc(
            government_states.nbytes
        )
        cuda_driver.memcpy_htod(
            self.government_action_indices_gpu_pycuda, government_action_indices
        )
        cuda_driver.memcpy_htod(self.government_actions_gpu_pycuda, government_actions)
        cuda_driver.memcpy_htod(self.government_rewards_gpu_pycuda, government_rewards)
        cuda_driver.memcpy_htod(
            self.government_states_checkpoint_gpu_pycuda, government_states
        )

    def __init_torch_data(self):

        __td = self.train_dict
        __ad = self.agents_dict

        batch_size = __td["batch_size"]
        num_consumers = __ad["num_consumers"]
        num_firms = __ad["num_firms"]
        num_governments = __ad["num_governments"]
        consumer_action_dim = __ad["consumer_action_dim"]
        consumer_state_dim = __ad["consumer_state_dim"]
        firm_state_dim = __ad["firm_state_dim"]
        government_state_dim = __ad["government_state_dim"]
        num_iters = int(self.world_dict["maxtime"])

        consumer_states_batch = torch.zeros(
            batch_size,
            num_iters,
            num_consumers,
            consumer_state_dim,
            dtype=torch.float32,
            device="cpu",
        )
        consumer_actions_single = torch.zeros(
            batch_size,
            num_consumers,
            num_firms + 1 + 1,
            dtype=torch.int32,
            device="cpu",
        )
        consumer_actions_batch = torch.zeros(
            batch_size,
            num_iters,
            num_consumers,
            num_firms + 1 + 1,
            dtype=torch.int32,
            device="cpu",
        )

        # auxiliary state info that is not part of observables.
        # currently just the realized consumption
        consumer_aux_batch = torch.zeros(
            batch_size,
            num_iters,
            num_consumers,
            num_firms,
            dtype=torch.float32,
            device="cpu",
        )

        consumer_rewards_batch = torch.zeros(
            batch_size, num_iters, num_consumers, dtype=torch.float32, device="cpu"
        )
        self.consumer_states_batch_gpu_tensor = consumer_states_batch.cuda()
        self.consumer_actions_batch_gpu_tensor = consumer_actions_batch.cuda()
        self.consumer_actions_index_single_gpu_tensor = consumer_actions_single.cuda()
        self.consumer_actions_single_gpu_tensor = torch.zeros(
            batch_size,
            num_consumers,
            consumer_action_dim,
            dtype=torch.float32,
            device="cpu",
        ).cuda()
        self.consumer_rewards_batch_gpu_tensor = consumer_rewards_batch.cuda()
        self.consumer_aux_batch_gpu_tensor = consumer_aux_batch.cuda()

        firm_states_batch = torch.zeros(
            batch_size,
            num_iters,
            num_firms,
            firm_state_dim,
            dtype=torch.float32,
            device="cpu",
        )
        firm_actions_batch = torch.zeros(
            batch_size, num_iters, num_firms, dtype=torch.int32, device="cpu"
        )
        firm_rewards_batch = torch.zeros(
            batch_size, num_iters, num_firms, dtype=torch.float32, device="cpu"
        )
        firm_aux_batch = torch.zeros(
            batch_size, num_iters, num_firms, dtype=torch.float32, device="cpu"
        )
        self.firm_states_batch = firm_states_batch.cuda()
        self.firm_actions_batch = firm_actions_batch.cuda()
        self.firm_rewards_batch = firm_rewards_batch.cuda()
        self.firm_aux_batch = firm_aux_batch.cuda()

        government_states_batch = torch.zeros(
            batch_size,
            num_iters,
            num_governments,
            government_state_dim,
            dtype=torch.float32,
            device="cpu",
        )
        government_actions_batch = torch.zeros(
            batch_size, num_iters, num_governments, dtype=torch.int32, device="cpu"
        )
        government_rewards_batch = torch.zeros(
            batch_size, num_iters, num_governments, dtype=torch.float32, device="cpu"
        )
        self.government_states_batch = government_states_batch.cuda()
        self.government_actions_batch = government_actions_batch.cuda()
        self.government_rewards_batch = government_rewards_batch.cuda()

    def __init_cuda_functions(self):

        __td = self.train_dict
        __ad = self.agents_dict
        __wd = self.world_dict

        if self.freeze_firms is not None:
            countfirmreward = 0
        else:
            countfirmreward = self.agents_dict["government_counts_firm_reward"]

        code, compiler_options = get_cuda_code(
            Path("cuda") / Path("firm_rbc.cu"),
            batchsize=__td["batch_size"],
            numconsumers=__ad["num_consumers"],
            numfirms=__ad["num_firms"],
            numgovernments=__ad["num_governments"],
            maxtime=__wd["maxtime"],
            # numactionsconsumer=__ad["consumer_num_actions"],
            numactionsconsumer=__ad["consumer_num_work_actions"],
            numactionsfirm=__ad["firm_num_actions"],
            numactionsgovernment=__ad["government_num_actions"],
            interestrate=__wd["interest_rate"],
            crra_param=__wd["crra_param"],
            shouldboostfirmreward=int(__td["should_boost_firm_reward"]),
            boostfirmrewardfactor=__td["boost_firm_reward_factor"],
            countfirmreward=countfirmreward,
            importerprice=__wd["importer_price"],
            importerquantity=__wd["importer_quantity"],
            laborfloor=__wd.get("labor_floor", 0.0),
            useimporter=__wd["use_importer"],
        )

        mod = SourceModule(code, options=compiler_options, no_extern_c=True)
        self.mod = mod

        # --------------------------------------------------------------------
        # Define Consumer actions -- maanged in Pytorch
        # --------------------------------------------------------------------
        self.consumption_action_tensor = torch.tensor(
            __ad["consumer_consumption_actions_array"].astype(_NP_DTYPE)
        ).cuda()
        self.work_action_tensor = torch.tensor(
            __ad["consumer_work_actions_array"].astype(_NP_DTYPE)
        ).cuda()

        # --------------------------------------------------------------------
        # Define Firm actions -- maanged in CUDA
        # --------------------------------------------------------------------
        firm_index_to_action_gpu, _ = mod.get_global("kFirmIndexToAction")
        cuda_driver.memcpy_htod(
            firm_index_to_action_gpu,
            __ad["firm_actions_array"].astype(_NP_DTYPE),
        )

        # --------------------------------------------------------------------
        # Define Govt actions -- maanged in CUDA
        # --------------------------------------------------------------------
        government_index_to_action_gpu, _ = mod.get_global("kGovernmentIndexToAction")
        cuda_driver.memcpy_htod(
            government_index_to_action_gpu,
            __ad["government_actions_array"].astype(_NP_DTYPE),
        )

        # --------------------------------------------------------------------
        # Get handles to CUDA methods
        # --------------------------------------------------------------------
        self.cuda_init_random = mod.get_function("CudaInitKernel")
        self.cuda_reset_env = mod.get_function("CudaResetEnv")
        self.cuda_sample_actions = mod.get_function(
            "CudaSampleFirmAndGovernmentActions"
        )
        self.cuda_step = mod.get_function("CudaStep")
        self.cuda_free_mem = mod.get_function("CudaFreeRand")

    def _update_consumer_actions_inplace(self):
        # call after consumer_actions_single is updated
        __ad = self.agents_dict

        # Add asserts when ``loading'' arrays
        # assert consumption_action_array.shape == (1, 1, 1)
        # assert len(consumption_action_array.shape) == 3

        num_firms = __ad["num_firms"]
        idx_hours = num_firms
        idx_which_firm = num_firms + 1
        for i in range(num_firms):

            consumption_actions_at_firm_i = (
                self.consumer_actions_index_single_gpu_tensor[..., i].to(torch.long)
            )

            self.consumer_actions_single_gpu_tensor[
                ..., i
            ] = self.consumption_action_tensor[
                consumption_actions_at_firm_i, :
            ].squeeze(
                dim=-1
            )

        consumer_hours_worked = self.consumer_actions_index_single_gpu_tensor[
            ..., idx_hours
        ].to(torch.long)

        self.consumer_actions_single_gpu_tensor[
            ..., num_firms
        ] = self.work_action_tensor[consumer_hours_worked, :].squeeze(dim=-1)

        self.consumer_actions_single_gpu_tensor[
            ..., num_firms + 1
        ] = self.consumer_actions_index_single_gpu_tensor[..., idx_which_firm]

    def sample_consumer_actions_and_store(self, consumer_probs_list):
        # Every consumer has A action heads, output as a list of tensors.
        # Sample from each of these lists and store the results.

        with torch.no_grad():
            for i, probs in enumerate(consumer_probs_list):
                dist = Categorical(probs)
                samples = dist.sample()
                self.consumer_actions_index_single_gpu_tensor[..., i] = samples

            self._update_consumer_actions_inplace()

    def consumers_will_train_this_episode(self, epi):
        __ad = self.agents_dict
        if "training_schedule_mod" in self.agents_dict:
            mod_val = epi % __ad["training_schedule_mod"]
            return mod_val <= __ad["consumer_mod_threshold"]
        if "consumer_training_list" in self.agents_dict:
            return interval_list_contains(__ad["consumer_training_list"], epi)
        if "train_consumers_every" in self.agents_dict:
            mod_val = epi % __ad["train_consumers_every"]
        else:
            mod_val = 0
        return epi >= self.agents_dict.get("consumer_training_start", 0) and (
            mod_val == 0
        )

    def firms_will_train_this_episode(self, epi):
        __ad = self.agents_dict
        if "training_schedule_mod" in self.agents_dict:
            mod_val = epi % __ad["training_schedule_mod"]
            return mod_val > __ad["consumer_mod_threshold"]
        if "firm_training_list" in self.agents_dict:
            return interval_list_contains(__ad["firm_training_list"], epi) and (
                self.freeze_firms is None
            )
        if "train_firms_every" in self.agents_dict:
            mod_val = epi % __ad["train_firms_every"]
        else:
            mod_val = 0
        return (
            (epi >= self.agents_dict.get("firm_training_start", 0))
            and (self.freeze_firms is None)
            and (mod_val == 0)
        )

    def governments_will_train_this_episode(self, epi):
        __ad = self.agents_dict
        if "government_training_list" in self.agents_dict:
            return interval_list_contains(__ad["government_training_list"], epi) and (
                self.freeze_govt is None
            )
        if "train_government_every" in self.agents_dict:
            mod_val = epi % self.agents_dict["train_government_every"]
        else:
            mod_val = 0
        return (
            (epi >= self.agents_dict.get("government_training_start", 0))
            and (self.freeze_govt is None)
            and (mod_val == 0)
        )

    def bestresponse_train(
        self, train_type, num_episodes, rollout_path, ep_str="latest", checkpoint=100
    ):
        # train one single type only
        # load all policies from state dict
        # reset all the environment stuff

        __td = self.train_dict
        __ad = self.agents_dict
        num_iters = int(self.world_dict["maxtime"])
        num_consumers = __ad["num_consumers"]
        num_firms = __ad["num_firms"]
        num_governments = __ad["num_governments"]
        num_agents = num_consumers + num_firms + num_governments
        block = (num_agents, 1, 1)
        grid = (__td["batch_size"], 1)

        seed_everything(__td["seed"])
        self.cuda_init_random(np.int32(__td["seed"]), block=block, grid=grid)

        # --------------------------------------------
        # Define Consumer policy + optimizers
        # --------------------------------------------
        lr = __td["lr"]

        consumer_expanded_size = size_after_digit_expansion(
            __ad["consumer_state_dim"],
            __ad["consumer_digit_dims"],
            __td["digit_representation_size"],
        )

        consumer_policy = IndependentPolicyNet(
            consumer_expanded_size,
            [__ad["consumer_num_consume_actions"]] * num_firms
            + [
                __ad["consumer_num_work_actions"],
                __ad["consumer_num_whichfirm_actions"],
            ],
            norm_consts=(
                torch.zeros(consumer_expanded_size).cuda(),  # don't center for now
                consumer_state_scaling_factors(self.cfg_dict),
            ),
        ).to("cuda")
        consumer_policy.load_state_dict(
            torch.load(
                rollout_path
                / Path("saved_models")
                / Path(f"consumer_policy_{ep_str}.pt")
            )
        )

        consumer_optim = torch.optim.Adam(consumer_policy.parameters(), lr=lr)
        firm_expanded_size = size_after_digit_expansion(
            __ad["firm_state_dim"],
            __ad["firm_digit_dims"],
            __td["digit_representation_size"],
        )
        firm_policy = PolicyNet(
            firm_expanded_size,
            __ad["firm_num_actions"],
            norm_consts=(
                torch.zeros(firm_expanded_size).cuda(),
                firm_state_scaling_factors(self.cfg_dict),
            ),
        ).to("cuda")

        firm_policy.load_state_dict(
            torch.load(
                rollout_path / Path("saved_models") / Path(f"firm_policy_{ep_str}.pt")
            )
        )

        firm_optim = torch.optim.Adam(firm_policy.parameters(), lr=lr)
        government_expanded_size = size_after_digit_expansion(
            __ad["government_state_dim"],
            __ad["government_digit_dims"],
            __td["digit_representation_size"],
        )

        government_policy = PolicyNet(
            government_expanded_size,
            __ad["government_num_actions"],
            norm_consts=(
                torch.zeros(government_expanded_size).cuda(),
                govt_state_scaling_factors(self.cfg_dict),
            ),
        ).to("cuda")

        government_policy.load_state_dict(
            torch.load(
                rollout_path
                / Path("saved_models")
                / Path(f"government_policy_{ep_str}.pt")
            )
        )

        government_optim = torch.optim.Adam(government_policy.parameters(), lr=lr)
        rewards = []

        agent_type_arrays = {
            "consumer": (
                self.consumer_states_batch_gpu_tensor,
                self.consumer_actions_batch_gpu_tensor,
                self.consumer_rewards_batch_gpu_tensor,
            ),
            "firm": (
                self.firm_states_batch,
                self.firm_actions_batch,
                self.firm_rewards_batch,
            ),
            "government": (
                self.government_states_batch,
                self.government_actions_batch,
                self.government_rewards_batch,
            ),
        }

        agent_action_arrays = {
            "consumer": __ad["consumer_work_actions_array"],
            "firm": __ad["firm_actions_array"],
            "government": __ad["government_actions_array"],
        }

        agent_aux_arrays = {
            "consumer": (self.consumer_aux_batch_gpu_tensor),
            "firm": (self.firm_aux_batch),
            "government": None,
        }

        pbar = tqdm(range(num_episodes))
        for epi in pbar:
            annealed_entropy_coef = 0.1  # later, do some computation to anneal this
            self.cuda_reset_env(
                CudaTensorHolder(self.consumer_states_gpu_tensor),
                CudaTensorHolder(self.firm_states_gpu_tensor),
                CudaTensorHolder(self.government_states_gpu_tensor),
                self.consumer_states_checkpoint_gpu_pycuda,
                self.firm_states_checkpoint_gpu_pycuda,
                self.government_states_checkpoint_gpu_pycuda,
                np.float32(1.0),
                block=block,
                grid=grid,
            )

            for _iter in range(num_iters):

                # ------------------------
                # Run policy and get probs
                # ------------------------
                with torch.no_grad():
                    # here, we must perform digit scaling
                    consumer_probs_list, _ = consumer_policy(
                        expand_to_digit_form(
                            self.consumer_states_gpu_tensor,
                            __ad["consumer_digit_dims"],
                            __td["digit_representation_size"],
                        )
                    )
                    firm_probs, _ = firm_policy(
                        expand_to_digit_form(
                            self.firm_states_gpu_tensor,
                            __ad["firm_digit_dims"],
                            __td["digit_representation_size"],
                        ),
                        actions_mask=None,
                    )
                    government_probs, _ = government_policy(
                        expand_to_digit_form(
                            self.government_states_gpu_tensor,
                            __ad["government_digit_dims"],
                            __td["digit_representation_size"],
                        ),
                        actions_mask=None,
                    )

                # ------------------------
                # Get action samples
                # ------------------------
                # Sample consumer actions using PyTorch here on GPU!
                self.sample_consumer_actions_and_store(consumer_probs_list)

                # Sample firms + govt actions using PyCUDA on GPU!
                self.cuda_sample_actions(
                    CudaTensorHolder(firm_probs),
                    self.firm_action_indices_gpu_pycuda,
                    self.firm_actions_gpu_pycuda,
                    CudaTensorHolder(government_probs),
                    self.government_action_indices_gpu_pycuda,
                    self.government_actions_gpu_pycuda,
                    block=block,
                    grid=grid,
                )

                # ------------------------
                # Step on GPU
                # ------------------------
                self.cuda_step(
                    CudaTensorHolder(
                        # size:  batches x n_consumers x consumer_state float
                        self.consumer_states_gpu_tensor
                    ),
                    CudaTensorHolder(
                        # size: batches x n_consumers x consumer_action_dim float
                        self.consumer_actions_single_gpu_tensor
                    ),
                    # size: batches x n_consumers x 1 float
                    self.consumer_rewards_gpu_pycuda,
                    CudaTensorHolder(
                        self.consumer_states_batch_gpu_tensor
                    ),  # size: batches x episode x n_consumers x consumer_state float
                    CudaTensorHolder(self.consumer_rewards_batch_gpu_tensor),
                    CudaTensorHolder(self.firm_states_gpu_tensor),
                    self.firm_action_indices_gpu_pycuda,
                    self.firm_actions_gpu_pycuda,
                    self.firm_rewards_gpu_pycuda,
                    CudaTensorHolder(self.firm_states_batch),
                    CudaTensorHolder(self.firm_actions_batch),
                    CudaTensorHolder(self.firm_rewards_batch),
                    CudaTensorHolder(self.government_states_gpu_tensor),
                    self.government_action_indices_gpu_pycuda,
                    self.government_actions_gpu_pycuda,
                    self.government_rewards_gpu_pycuda,
                    CudaTensorHolder(self.government_states_batch),
                    CudaTensorHolder(self.government_actions_batch),
                    CudaTensorHolder(self.government_rewards_batch),
                    CudaTensorHolder(self.consumer_aux_batch_gpu_tensor),
                    CudaTensorHolder(self.firm_aux_batch),
                    np.int32(_iter),
                    block=block,
                    grid=grid,
                )
                self.consumer_actions_batch_gpu_tensor[
                    :, _iter, :, :
                ] = self.consumer_actions_index_single_gpu_tensor
            update_government_rewards(
                self.government_rewards_batch,
                self.consumer_rewards_batch_gpu_tensor,
                self.firm_rewards_batch,
                self.cfg_dict,
            )
            if train_type == "consumer":
                consumer_reward_scale = self.agents_dict.get(
                    "consumer_reward_scale", 1.0
                )
                consumer_policy_gradient_step(
                    consumer_policy,
                    expand_to_digit_form(
                        self.consumer_states_batch_gpu_tensor,
                        __ad["consumer_digit_dims"],
                        __td["digit_representation_size"],
                    ),
                    self.consumer_actions_batch_gpu_tensor,
                    self.consumer_rewards_batch_gpu_tensor,
                    consumer_optim,
                    __td["gamma"],
                    entropy_val=annealed_entropy_coef * __td["entropy"],
                    value_loss_weight=__td["value_loss_weight"],
                    reward_scale=consumer_reward_scale,
                    clip_grad_norm=self.train_dict.get("clip_grad_norm", None),
                )
                rewards.append(self.consumer_rewards_batch_gpu_tensor.mean().item())
            elif train_type == "firm":
                firm_reward_scale = self.agents_dict.get("firm_reward_scale", 1.0)
                policy_gradient_step(
                    firm_policy,
                    expand_to_digit_form(
                        self.firm_states_batch,
                        __ad["firm_digit_dims"],
                        __td["digit_representation_size"],
                    ),
                    self.firm_actions_batch,
                    self.firm_rewards_batch,
                    firm_optim,
                    __td["gamma"],
                    entropy_val=annealed_entropy_coef * __td["entropy"],
                    value_loss_weight=__td["value_loss_weight"],
                    actions_mask=None,
                    reward_scale=firm_reward_scale,
                    clip_grad_norm=self.train_dict.get("clip_grad_norm", None),
                )
                rewards.append(self.firm_rewards_batch.mean().item())
            elif train_type == "government":
                government_reward_scale = self.agents_dict.get(
                    "government_reward_scale", 1.0
                )
                policy_gradient_step(
                    government_policy,
                    expand_to_digit_form(
                        self.government_states_batch,
                        __ad["government_digit_dims"],
                        __td["digit_representation_size"],
                    ),
                    self.government_actions_batch,
                    self.government_rewards_batch,
                    government_optim,
                    __td["gamma"],
                    entropy_val=annealed_entropy_coef * __td["entropy"],
                    value_loss_weight=__td["value_loss_weight"],
                    actions_mask=None,
                    reward_scale=government_reward_scale,
                    clip_grad_norm=self.train_dict.get("clip_grad_norm", None),
                )
                rewards.append(self.government_rewards_batch.mean().item())
            pbar.set_postfix({"reward": rewards[-1]})
            if (epi % checkpoint) == 0:
                # save policy every checkpoint steps
                save_policy_parameters(
                    str(Path(self.save_dir) / f"br{train_type}"),
                    epi,
                    consumer_policy,
                    firm_policy,
                    government_policy,
                    self.freeze_firms,
                    self.freeze_govt,
                )
                save_dense_log(
                    str(Path(self.save_dir) / f"br{train_type}"),
                    epi,
                    agent_type_arrays,
                    agent_action_arrays,
                    agent_aux_arrays,
                )

        print(
            f"{train_type}: starting reward {rewards[0]}, "
            f"ending reward {rewards[-1]}, "
            f"improvement in reward after {num_episodes}: {rewards[-1] - rewards[0]}"
        )

        self.cuda_free_mem(block=block, grid=grid)
        return rewards

    def train(self):

        __td = self.train_dict
        __ad = self.agents_dict

        # Create logdir
        os.makedirs(__td["save_dir"], exist_ok=True)

        # Constants
        num_iters = int(self.world_dict["maxtime"])
        num_consumers = __ad["num_consumers"]
        num_firms = __ad["num_firms"]
        num_governments = __ad["num_governments"]
        num_agents = num_consumers + num_firms + num_governments

        # CUDA params: defines data shape on the GPU
        block = (num_agents, 1, 1)
        grid = (__td["batch_size"], 1)

        # Set seeds
        seed_everything(__td["seed"])
        self.cuda_init_random(np.int32(__td["seed"]), block=block, grid=grid)

        # --------------------------------------------
        # Define Consumer policy + optimizers
        # --------------------------------------------
        lr = __td["lr"]

        consumer_expanded_size = size_after_digit_expansion(
            __ad["consumer_state_dim"],
            __ad["consumer_digit_dims"],
            __td["digit_representation_size"],
        )

        consumer_policy = IndependentPolicyNet(
            consumer_expanded_size,
            [__ad["consumer_num_consume_actions"]] * num_firms
            + [
                __ad["consumer_num_work_actions"],
                __ad["consumer_num_whichfirm_actions"],
            ],
            norm_consts=(
                torch.zeros(consumer_expanded_size).cuda(),  # don't center for now
                consumer_state_scaling_factors(self.cfg_dict),
            ),
        ).to("cuda")

        consumer_optim = torch.optim.Adam(
            consumer_policy.parameters(),
            lr=lr * self.agents_dict.get("consumer_lr_multiple", 1.0),
        )

        # --------------------------------------------
        # Define Firm policy + optimizers
        # --------------------------------------------
        firm_expanded_size = size_after_digit_expansion(
            __ad["firm_state_dim"],
            __ad["firm_digit_dims"],
            __td["digit_representation_size"],
        )

        if self.freeze_firms is not None:
            firm_policy = DeterministicPolicy(
                firm_expanded_size,
                __ad["firm_num_actions"],
                self.freeze_firms,
            )
            firm_optim = NoOpOptimizer()
        else:
            firm_policy = PolicyNet(
                firm_expanded_size,
                __ad["firm_num_actions"],
                norm_consts=(
                    torch.zeros(firm_expanded_size).cuda(),
                    firm_state_scaling_factors(self.cfg_dict),
                ),
            ).to("cuda")

            firm_optim = torch.optim.Adam(
                firm_policy.parameters(),
                lr=lr * self.agents_dict.get("firm_lr_multiple", 1.0),
            )

        # --------------------------------------------
        # Define Government policy + optimizers
        # --------------------------------------------
        government_expanded_size = size_after_digit_expansion(
            __ad["government_state_dim"],
            __ad["government_digit_dims"],
            __td["digit_representation_size"],
        )

        if self.freeze_govt is not None:
            government_policy = DeterministicPolicy(
                government_expanded_size,
                __ad["government_num_actions"],
                self.freeze_govt,
            )
            government_optim = NoOpOptimizer()
        else:
            government_policy = PolicyNet(
                government_expanded_size,
                __ad["government_num_actions"],
                norm_consts=(
                    torch.zeros(government_expanded_size).cuda(),
                    govt_state_scaling_factors(self.cfg_dict),
                ),
            ).to("cuda")
            government_optim = torch.optim.Adam(
                government_policy.parameters(),
                lr=lr * self.agents_dict.get("government_lr_multiple", 1.0),
            )

        # --------------------------------------------
        # Logging
        # --------------------------------------------
        # For looking up GPU tensors
        # --------------------------------------------
        agent_type_arrays = {
            "consumer": (
                self.consumer_states_batch_gpu_tensor,
                self.consumer_actions_batch_gpu_tensor,
                self.consumer_rewards_batch_gpu_tensor,
            ),
            "firm": (
                self.firm_states_batch,
                self.firm_actions_batch,
                self.firm_rewards_batch,
            ),
            "government": (
                self.government_states_batch,
                self.government_actions_batch,
                self.government_rewards_batch,
            ),
        }

        agent_action_arrays = {
            "consumer": __ad["consumer_work_actions_array"],
            "firm": __ad["firm_actions_array"],
            "government": __ad["government_actions_array"],
        }

        agent_aux_arrays = {
            "consumer": (self.consumer_aux_batch_gpu_tensor),
            "firm": (self.firm_aux_batch),
            "government": None,
        }

        # --------------------------------------------
        # Training policy XYZ starts at which step?
        # --------------------------------------------
        firm_no_ponzi_coef = self.agents_dict.get("firm_noponzi_start", 0.0)
        consumer_no_ponzi_coef = self.agents_dict.get("consumer_noponzi_start", 0.0)
        lagr_num_steps = self.train_dict.get("lagr_num_steps", 1)

        firm_training_start = self.agents_dict.get("firm_training_start", 0)
        consumer_training_start = self.agents_dict.get("consumer_training_start", 0)
        government_training_start = self.agents_dict.get("government_training_start", 0)

        firm_action_start = self.agents_dict.get("firm_begin_anneal_action", 0)
        government_action_start = self.agents_dict.get(
            "government_begin_anneal_action", 0
        )

        # --------------------------------------------
        # Training loop
        # --------------------------------------------
        if self.train_dict.get("infinite_episodes", False):
            epi_iterator = itertools.count(0, 1)
        else:
            epi_iterator = range(__td["num_episodes"])

        final_epi = None
        for epi in tqdm(epi_iterator):

            firm_actions_mask = firm_action_mask(
                self.cfg_dict,
                max(epi - firm_action_start, 0),
            )
            government_actions_mask = government_action_mask(
                self.cfg_dict,
                max(epi - government_action_start, 0),
            )
            theta_coef = compute_theta_coef(self.cfg_dict, epi)

            # Reset environment for all agents
            self.cuda_reset_env(
                CudaTensorHolder(self.consumer_states_gpu_tensor),
                CudaTensorHolder(self.firm_states_gpu_tensor),
                CudaTensorHolder(self.government_states_gpu_tensor),
                self.consumer_states_checkpoint_gpu_pycuda,
                self.firm_states_checkpoint_gpu_pycuda,
                self.government_states_checkpoint_gpu_pycuda,
                theta_coef,
                block=block,
                grid=grid,
            )

            # Learning Loop
            for _iter in range(num_iters):

                # ------------------------
                # Run policy and get probs
                # ------------------------
                with torch.no_grad():
                    # here, we must perform digit scaling
                    consumer_probs_list, _ = consumer_policy(
                        expand_to_digit_form(
                            self.consumer_states_gpu_tensor,
                            __ad["consumer_digit_dims"],
                            __td["digit_representation_size"],
                        )
                    )
                    firm_probs, _ = firm_policy(
                        expand_to_digit_form(
                            self.firm_states_gpu_tensor,
                            __ad["firm_digit_dims"],
                            __td["digit_representation_size"],
                        ),
                        actions_mask=firm_actions_mask,
                    )
                    government_probs, _ = government_policy(
                        expand_to_digit_form(
                            self.government_states_gpu_tensor,
                            __ad["government_digit_dims"],
                            __td["digit_representation_size"],
                        ),
                        actions_mask=government_actions_mask,
                    )

                # ------------------------
                # Get action samples
                # ------------------------
                # Sample consumer actions using PyTorch here on GPU!
                self.sample_consumer_actions_and_store(consumer_probs_list)

                # Sample firms + govt actions using PyCUDA on GPU!
                self.cuda_sample_actions(
                    CudaTensorHolder(firm_probs),
                    self.firm_action_indices_gpu_pycuda,
                    self.firm_actions_gpu_pycuda,
                    CudaTensorHolder(government_probs),
                    self.government_action_indices_gpu_pycuda,
                    self.government_actions_gpu_pycuda,
                    block=block,
                    grid=grid,
                )

                # ------------------------
                # Step on GPU
                # ------------------------
                self.cuda_step(
                    CudaTensorHolder(
                        # size:  batches x n_consumers x consumer_state float
                        self.consumer_states_gpu_tensor
                    ),
                    CudaTensorHolder(
                        # size: batches x n_consumers x consumer_action_dim float
                        self.consumer_actions_single_gpu_tensor
                    ),
                    # size: batches x n_consumers x 1 float
                    self.consumer_rewards_gpu_pycuda,
                    CudaTensorHolder(
                        # size: batches x episode x n_consumers x consumer_state float
                        self.consumer_states_batch_gpu_tensor
                    ),
                    CudaTensorHolder(self.consumer_rewards_batch_gpu_tensor),
                    CudaTensorHolder(self.firm_states_gpu_tensor),
                    self.firm_action_indices_gpu_pycuda,
                    self.firm_actions_gpu_pycuda,
                    self.firm_rewards_gpu_pycuda,
                    CudaTensorHolder(self.firm_states_batch),
                    CudaTensorHolder(self.firm_actions_batch),
                    CudaTensorHolder(self.firm_rewards_batch),
                    CudaTensorHolder(self.government_states_gpu_tensor),
                    self.government_action_indices_gpu_pycuda,
                    self.government_actions_gpu_pycuda,
                    self.government_rewards_gpu_pycuda,
                    CudaTensorHolder(self.government_states_batch),
                    CudaTensorHolder(self.government_actions_batch),
                    CudaTensorHolder(self.government_rewards_batch),
                    CudaTensorHolder(self.consumer_aux_batch_gpu_tensor),
                    CudaTensorHolder(self.firm_aux_batch),
                    np.int32(_iter),
                    block=block,
                    grid=grid,
                )
                self.consumer_actions_batch_gpu_tensor[
                    :, _iter, :, :
                ] = self.consumer_actions_index_single_gpu_tensor

            # ------------------------
            # Add penalty for no-Ponzi
            # ------------------------
            add_penalty_for_no_ponzi(
                self.firm_states_gpu_tensor,
                self.firm_rewards_batch,
                __ad["global_state_dim"],
                penalty_coef=firm_no_ponzi_coef,
            )
            add_penalty_for_no_ponzi(
                self.consumer_states_gpu_tensor,
                self.consumer_rewards_batch_gpu_tensor,
                __ad["global_state_dim"],
                penalty_coef=consumer_no_ponzi_coef,
                penalty_scale=__ad["consumer_penalty_scale"],
            )

            # add government rewards -- sum of consumer rewards
            update_government_rewards(
                self.government_rewards_batch,
                self.consumer_rewards_batch_gpu_tensor,
                self.firm_rewards_batch,
                self.cfg_dict,
            )

            # Save dense logs
            # ------------------------
            if (epi % __td["save_model_every"]) == 0:
                save_policy_parameters(
                    self.save_dir,
                    epi,
                    consumer_policy,
                    firm_policy,
                    government_policy,
                    self.freeze_firms,
                    self.freeze_govt,
                )
            if (epi % self.save_dense_every) == 0:
                save_dense_log(
                    self.save_dir,
                    epi,
                    agent_type_arrays,
                    agent_action_arrays,
                    agent_aux_arrays,
                )

            # --------------------------------
            # Curriculum: Train Consumers
            # --------------------------------
            if self.consumers_will_train_this_episode(epi):
                consumer_entropy_coef = anneal_entropy_coef(
                    self.agents_dict.get("consumer_anneal_entropy", None),
                    epi - consumer_training_start,
                )
                consumer_reward_scale = self.agents_dict.get(
                    "consumer_reward_scale", 1.0
                )
                if __td["use_ppo"]:
                    consumer_ppo_step(
                        consumer_policy,
                        expand_to_digit_form(
                            self.consumer_states_batch_gpu_tensor,
                            __ad["consumer_digit_dims"],
                            __td["digit_representation_size"],
                        ),
                        self.consumer_actions_batch_gpu_tensor,
                        self.consumer_rewards_batch_gpu_tensor,
                        consumer_optim,
                        __td["gamma"],
                        entropy_val=consumer_entropy_coef * __td["entropy"],
                        value_loss_weight=__td["value_loss_weight"],
                        reward_scale=consumer_reward_scale,
                        ppo_num_updates=__td["ppo_num_updates"],
                        clip_param=__td["ppo_clip_param"],
                        clip_grad_norm=self.train_dict.get("clip_grad_norm", None),
                    )
                else:
                    consumer_policy_gradient_step(
                        consumer_policy,
                        expand_to_digit_form(
                            self.consumer_states_batch_gpu_tensor,
                            __ad["consumer_digit_dims"],
                            __td["digit_representation_size"],
                        ),
                        self.consumer_actions_batch_gpu_tensor,
                        self.consumer_rewards_batch_gpu_tensor,
                        consumer_optim,
                        __td["gamma"],
                        entropy_val=consumer_entropy_coef * __td["entropy"],
                        value_loss_weight=__td["value_loss_weight"],
                        reward_scale=consumer_reward_scale,
                        clip_grad_norm=self.train_dict.get("clip_grad_norm", None),
                    )
                    if (epi % lagr_num_steps) == 0:
                        consumer_no_ponzi_coef = update_penalty_coef(
                            self.consumer_states_gpu_tensor,
                            __ad["global_state_dim"],
                            consumer_no_ponzi_coef,
                            penalty_step_size=__ad["consumer_noponzi_eta"],
                            penalty_scale=__ad["consumer_penalty_scale"],
                        )
            else:
                pass

            # --------------------------------
            # Curriculum: Train Firms
            # --------------------------------
            if self.firms_will_train_this_episode(epi):
                firm_entropy_coef = anneal_entropy_coef(
                    self.agents_dict.get("firm_anneal_entropy", None),
                    epi - firm_training_start,
                )
                firm_reward_scale = self.agents_dict.get("firm_reward_scale", 1.0)
                if __td["use_ppo"]:
                    ppo_step(
                        firm_policy,
                        expand_to_digit_form(
                            self.firm_states_batch,
                            __ad["firm_digit_dims"],
                            __td["digit_representation_size"],
                        ),
                        self.firm_actions_batch,
                        self.firm_rewards_batch,
                        firm_optim,
                        __td["gamma"],
                        entropy_val=firm_entropy_coef * __td["entropy"],
                        value_loss_weight=__td["value_loss_weight"],
                        actions_mask=firm_actions_mask,
                        reward_scale=firm_reward_scale,
                        ppo_num_updates=__td["ppo_num_updates"],
                        clip_param=__td["ppo_clip_param"],
                        clip_grad_norm=self.train_dict.get("clip_grad_norm", None),
                    )
                else:
                    policy_gradient_step(
                        firm_policy,
                        expand_to_digit_form(
                            self.firm_states_batch,
                            __ad["firm_digit_dims"],
                            __td["digit_representation_size"],
                        ),
                        self.firm_actions_batch,
                        self.firm_rewards_batch,
                        firm_optim,
                        __td["gamma"],
                        entropy_val=firm_entropy_coef * __td["entropy"],
                        value_loss_weight=__td["value_loss_weight"],
                        actions_mask=firm_actions_mask,
                        reward_scale=firm_reward_scale,
                        clip_grad_norm=self.train_dict.get("clip_grad_norm", None),
                    )

                if (epi % lagr_num_steps) == 0:
                    firm_no_ponzi_coef = update_penalty_coef(
                        self.firm_states_gpu_tensor,
                        __ad["global_state_dim"],
                        firm_no_ponzi_coef,
                        penalty_step_size=__ad["firm_noponzi_eta"],
                    )
            else:
                pass

            # --------------------------------
            # Curriculum: Train Governments
            # --------------------------------
            if self.governments_will_train_this_episode(epi):
                government_entropy_coef = anneal_entropy_coef(
                    self.agents_dict.get("govt_anneal_entropy", None),
                    epi - government_training_start,
                )
                government_reward_scale = self.agents_dict.get(
                    "government_reward_scale", 1.0
                )
                if __td["use_ppo"]:
                    ppo_step(
                        government_policy,
                        expand_to_digit_form(
                            self.government_states_batch,
                            __ad["government_digit_dims"],
                            __td["digit_representation_size"],
                        ),
                        self.government_actions_batch,
                        self.government_rewards_batch,
                        government_optim,
                        __td["gamma"],
                        entropy_val=government_entropy_coef * __td["entropy"],
                        value_loss_weight=__td["value_loss_weight"],
                        actions_mask=government_actions_mask,
                        reward_scale=government_reward_scale,
                        ppo_num_updates=__td["ppo_num_updates"],
                        clip_param=__td["ppo_clip_param"],
                        clip_grad_norm=self.train_dict.get("clip_grad_norm", None),
                    )
                else:
                    policy_gradient_step(
                        government_policy,
                        expand_to_digit_form(
                            self.government_states_batch,
                            __ad["government_digit_dims"],
                            __td["digit_representation_size"],
                        ),
                        self.government_actions_batch,
                        self.government_rewards_batch,
                        government_optim,
                        __td["gamma"],
                        entropy_val=government_entropy_coef * __td["entropy"],
                        value_loss_weight=__td["value_loss_weight"],
                        actions_mask=government_actions_mask,
                        reward_scale=government_reward_scale,
                        clip_grad_norm=self.train_dict.get("clip_grad_norm", None),
                    )
            else:
                pass

            # Store the value of the final episode
            final_epi = epi

        # ------------------------------------------------------------------
        # Post-Training (may not reach this with an infinite training loop!)
        # Save FINAL dense log.
        # ------------------------------------------------------------------
        save_dense_log(
            self.save_dir,
            "final",
            agent_type_arrays,
            agent_action_arrays,
            agent_aux_arrays,
        )
        save_policy_parameters(
            self.save_dir,
            final_epi,
            consumer_policy,
            firm_policy,
            government_policy,
            self.freeze_firms,
            self.freeze_govt,
        )

        # ------------------------------------------------------------------
        # Clean up
        # ------------------------------------------------------------------

        self.cuda_free_mem(block=block, grid=grid)


class CudaTensorHolder(pycuda.driver.PointerHolderBase):
    """
    A class that facilitates casting tensors to pointers.
    """

    def __init__(self, t):
        super().__init__()
        self.t = t
        self.gpudata = t.data_ptr()

    def get_pointer(self):
        return self.t.data_ptr()
