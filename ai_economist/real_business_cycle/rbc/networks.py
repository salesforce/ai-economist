# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import torch
import torch.nn.functional as F
from torch import nn


class IndependentPolicyNet(nn.Module):
    """
    Represents a policy network with separate heads for different types of actions.
    Thus, the resulting policy will take the form
    $pi(a | s) = pi_1(a_1 | s) pi_2(a_2 | s)...$
    """

    def __init__(self, state_size, action_size_list, norm_consts=None):
        super().__init__()

        self.state_size = state_size
        self.action_size_list = action_size_list
        if norm_consts is not None:
            self.norm_center, self.norm_scale = norm_consts
        else:
            self.norm_center = torch.zeros(self.state_size).cuda()
            self.norm_scale = torch.ones(self.state_size).cuda()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        # policy network head
        self.action_heads = nn.ModuleList(
            [nn.Linear(128, action_size) for action_size in action_size_list]
        )
        # value network head
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        assert x.shape[-1] == self.state_size  # Check if the last dimension matches

        # Normalize the model input
        new_shape = tuple(1 for _ in x.shape[:-1]) + (x.shape[-1],)
        view_center = self.norm_center.view(new_shape)
        view_scale = self.norm_scale.view(new_shape)
        x = (x - view_center) / view_scale

        # Feed forward
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        probs = [F.softmax(action_head(x), dim=-1) for action_head in self.action_heads]
        vals = self.fc4(x)
        return probs, vals


class PolicyNet(nn.Module):
    """
    The policy network class to output acton probabilities and the value function.
    """

    def __init__(self, state_size, action_size, norm_consts=None):
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        if norm_consts is not None:
            self.norm_center, self.norm_scale = norm_consts
        else:
            self.norm_center = torch.zeros(self.state_size).cuda()
            self.norm_scale = torch.ones(self.state_size).cuda()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        # policy network head
        self.fc3 = nn.Linear(128, action_size)
        # value network head
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x, actions_mask=None):
        # here, the action mask should be large negative constants for actions
        # that shouldn't be allowed.
        new_shape = tuple(1 for _ in x.shape[:-1]) + (x.shape[-1],)
        view_center = self.norm_center.view(new_shape)
        view_scale = self.norm_scale.view(new_shape)
        x = (x - view_center) / view_scale
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if actions_mask is not None:
            probs = F.softmax(self.fc3(x) + actions_mask, dim=-1)
        else:
            probs = F.softmax(self.fc3(x), dim=-1)
        vals = self.fc4(x)
        return probs, vals


class DeterministicPolicy:
    """
    A policy class that outputs deterministic actions.
    """

    def __init__(self, state_size, action_size, action_choice):
        self.state_size = state_size
        self.action_size = action_size
        self.action_choice = action_choice
        self.actions_out = torch.zeros(action_size, device="cuda")
        self.actions_out[self.action_choice] = 1.0

    def __call__(self, x, actions_mask=None):
        return self.forward(x)

    def forward(self, x):
        # output enough copies of the delta function
        # distribution of the right size given x
        x_batch_shapes = x.shape[:-1]
        repeat_vals = x_batch_shapes + (1,)
        return self.actions_out.repeat(*repeat_vals), None
