# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import torch


def dict_merge(dct, merge_dct):
    """Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.items():
        # dct does not have k yet. Add it with value v.
        if (k not in dct) and (not isinstance(dct, list)):
            dct[k] = v
        else:
            # dct[k] and merge_dict[k] are both dictionaries. Recurse.
            if isinstance(dct[k], (dict, list)) and isinstance(v, dict):
                dict_merge(dct[k], merge_dct[k])
            else:
                # dct[k] and merge_dict[k] are both tuples or lists.
                if isinstance(dct[k], (tuple, list)) and isinstance(v, (tuple, list)):
                    # They don't match. Overwrite with v.
                    if len(dct[k]) != len(v):
                        dct[k] = v
                    else:
                        for i, (d_val, v_val) in enumerate(zip(dct[k], v)):
                            if isinstance(d_val, dict) and isinstance(v_val, dict):
                                dict_merge(d_val, v_val)
                            else:
                                dct[k][i] = v_val
                else:
                    dct[k] = v


def min_max_consumer_budget_delta(hparams_dict):
    # largest single round changes
    max_wage = hparams_dict["agents"]["max_possible_wage"]
    max_hours = hparams_dict["agents"]["max_possible_hours_worked"]
    max_price = hparams_dict["agents"]["max_possible_price"]
    max_singlefirm_consumption = hparams_dict["agents"]["max_possible_consumption"]
    num_firms = hparams_dict["agents"]["num_firms"]

    min_budget = (
        -max_singlefirm_consumption * max_price * num_firms
    )  # negative budget from consuming only
    max_budget = max_hours * max_wage * num_firms  # positive budget from only working
    return min_budget, max_budget


def min_max_stock_delta(hparams_dict):
    # for now, assuming 1.0 capital
    max_hours = hparams_dict["agents"]["max_possible_hours_worked"]
    max_singlefirm_consumption = hparams_dict["agents"]["max_possible_consumption"]
    alpha = hparams_dict["world"]["production_alpha"]
    if isinstance(alpha, str):
        alpha = 0.8
    num_consumers = hparams_dict["agents"]["num_consumers"]
    max_delta = (max_hours * num_consumers) ** alpha
    min_delta = -max_singlefirm_consumption * num_consumers
    return min_delta, max_delta


def min_max_firm_budget(hparams_dict):
    max_wage = hparams_dict["agents"]["max_possible_wage"]
    max_hours = hparams_dict["agents"]["max_possible_hours_worked"]
    max_singlefirm_consumption = hparams_dict["agents"]["max_possible_consumption"]
    num_consumers = hparams_dict["agents"]["num_consumers"]
    max_price = hparams_dict["agents"]["max_possible_price"]
    max_delta = max_singlefirm_consumption * max_price * num_consumers
    min_delta = -max_hours * max_wage * num_consumers
    return min_delta, max_delta


def expand_to_digit_form(x, dims_to_expand, max_digits):
    # first split x up
    requires_grad = (
        x.requires_grad
    )  # don't want to backprop through these ops, but do want
    # gradients if x had them
    with torch.no_grad():
        tensor_pieces = []
        expanded_digit_shape = x.shape[:-1] + (max_digits,)
        for i in range(x.shape[-1]):
            if i not in dims_to_expand:
                tensor_pieces.append(x[..., i : i + 1])
            else:
                digit_entries = torch.zeros(expanded_digit_shape, device=x.device)
                for j in range(max_digits):
                    digit_entries[..., j] = (x[..., i] % (10 ** (j + 1))) / (
                        10 ** (j + 1)
                    )
                tensor_pieces.append(digit_entries)

    output = torch.cat(tensor_pieces, dim=-1)
    output.requires_grad_(requires_grad)
    return output


def size_after_digit_expansion(existing_size, dims_to_expand, max_digits):
    num_expanded = len(dims_to_expand)
    # num non expanded digits, + all the expanded ones
    return (existing_size - num_expanded) + (max_digits * num_expanded)
