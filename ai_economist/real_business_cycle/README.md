# Real Business Cycle (RBC)
This directory implements a **Real-Business-Cycle** (RBC) simulation with many heterogeneous, interacting strategic agents of various types, such as **consumers, firms, and the government**. For details, please refer to this paper "Finding General Equilibria in Many-Agent Economic Simulations using Deep Reinforcement Learning (ArXiv link forthcoming)". We also provide training code that uses deep multi-agent reinforcement learning to determine optimal economic policies and dynamics in these many agent environments. Below are instructions required to launch the training runs.

**Note: The experiments require a GPU to run!**

## Dependencies

- torch>=1.9.0
- pycuda==2021.1
- matplotlib==3.2.1

## Running Local Jobs
To run a hyperparameter sweep of jobs on a local machine, use (see file for command line arguments and hyperparameter sweep dictionaries)

```
python train_multi_exps.py
```

## Configuration Dictionaries

Configuration dictionaries are currently specified in Python code, and then written as `hparams.yaml` in the job directory. For examples, see the file `constants.py`. The dictionaries contain "agents", "world", and "train" dictionaries which contain various hyperparameters.

## Hyperparameter Sweeps

The files `train_multi_exps.py` allow hyperparameter sweeps. These are specified in `*_param_sweeps` dictionaries in the file. For each hyperparameter, specify a list of one or more choices. The Cartesian product of all choices will be used.

## Approximate Best Response Training

To run a single approximate best-response (BR) training job on checkpoint policies, run `python train_bestresponse.py ROLLOUT_DIR NUM_EPISODES_TO_TRAIN --ep-strs ep1 ep2 --agent-type all`. The `--ep-strs` argument specifies which episodes to run on (for example, policies from episode 0, 10000, and 200000). These must be episodes for which policies were saved. It is possible to specify a single agent type.


## What Will Be Saved?

A large amount of data will be saved -- one can set hyperparamter `train.save_dense_every` in the configuration dictionary (`hparams.yaml`/`constants.py`) to reduce this.

At the top level, an experiment directory stores the results of many runs in a hyperparameter sweep. Example structure:

```
experiment/experimentname/
    rollout-999999-99999/
        brconsumer/
            ...
        brfirm/
            episode_XXXX_consumer.npz
            episode_XXXX_government.npz
            episode_XXXX_firm.npz
            saved_models/
                consumer_policy_XXX.pt
                firm_policy_XXX.pt
                government_policy_XXX.pt.
        brgovernment/
            ...
        hparams.yaml
        action_arrays.pickle
        episode_XXXX_consumer.npz
        episode_XXXX_government.npz
        episode_XXXX_firm.npz
        saved_models/
            consumer_policy_XXX.pt
            firm_policy_XXX.pt
            government_policy_XXX.pt.

    rollout-777777-77777/
        ...
```

Files:

`rollout-XXXXXX-XXX`: subdirectory containing all output for a single run.

`hparams.yaml`: configuration dictionary with hyperparameters

`action_arrays.pickle`: contains saved action arrays (allowing mapping action indices to the actual action, e.g. index 1 is price 1000.0, etc.)

`episode_XXXX_AGENTTYPE.npz`: Contains dense rollouts stored as the output of a numpy.savez call. When loaded, can be treated like a dictionary of numpy arrays.  Has keys: `['states', 'actions', 'rewards', 'action_array', 'aux_array']` (view keys by using `.files`). `states`, `actions`, `rewards`, and `aux_array` all refer to saved copies of CUDA arrays (described below). `action_array` is a small array mapping action indices to the actual action.

`saved_models/AGENTTYPE_policy_XXX.pt`: a saved PyTorch state dict of the policy network, after episode XXX.

## Structure Of Arrays

`states` for any given agent type is an array storing observed states. It has shape `batch_size, ep_length, num_agents, agent_total_state_dim`.

`actions` is an array consisting of the action _indices_ (integers). For firms and government, it is of shape `batch_size, ep_length, num_agents`. For consumers, it is of shape `batch_size, ep_length, num_agents, num_action_heads`.

`rewards` stores total rewards, and is of shape `batch_size, ep_length, num_agents`.

The `aux_array` stores additional information and may differ per agent type. The consumer `aux_array` stores _actual_ consumption of each firm's good (as opposed to attempted consumption). The firm `aux_array` stores the amount bought by the export market.

## State Array Layout:

States observed by each agent consist of a global state, plus additional state dimensions per agent.

Global state: total dimension 4 * num_firms + 2 + 1
- prices: 1 per firm
- wages: 1 per firm
- inventories: 1 per firm
- overdemanded flag: 1 per firm
- time

Consumer additional state variables: total dimension global state + 2
- budget
- theta

Firm additional state variables: total dimension global state + 3 + num_firms
- budget
- capital
- production alpha
- one-hot representation identifying which firm

## What Gets Loaded And Written By BR Code?

The best response code loads in the `hparams.yaml` file, and the policies at a given time step (i.e. `saved_models/...policy_XXX.pt`). It then trains one of the policies while keeping the others fixed. Results are written to directories `brfirm`, `brconsumer`, `brgovernment` and contain dense rollouts and saved policy checkpoints, but from the best response training.

## Which Hyperparameters Are Managed And Where?

Initial values of state variables (budgets, initial wages, levels of capital, and so on) are set by the code in the method `__init_cuda_data_structs`. Some of these can be controlled from the hyperparameter dict; others are currently hardcoded.

Other hyperparameters are specified in the configuration dictionary.

Finally, the technology parameter (A) of the production function is currently hardcoded in the function call in `rbc/cuda/firm_rbc.cu`.
