# Foundation: An Economic Simulation Framework

This repo contains an implementation of Foundation, a framework for flexible, modular, and composable environments that **model socio-economic behaviors and dynamics in a society with both agents and governments**.

Foundation provides a [Gym](https://gym.openai.com/)-style API:

- `reset`: resets the environment's state and returns the observation.
- `step`: advances the environment by one timestep, and returns the tuple *(observation, reward, done, info)*.

This simulation can be used in conjunction with reinforcement learning to learn optimal economic policies, as detailed in:

**[The AI Economist: Improving Equality and Productivity with AI-Driven Tax Policies](https://arxiv.org/abs/2004.13332)**,
*Stephan Zheng, Alexander Trott, Sunil Srinivasa, Nikhil Naik, Melvin Gruesbeck, David C. Parkes, Richard Socher.*

## Installation Instructions

To get started, you'll need to have Python 3.6+ installed.

### Using pip

Simply use the Python package manager:

```python
pip install ai-economist
```

### Installing from Source

1. Clone this repository to your local machine:

    ```
    git clone www.github.com/swarnabha13/ai-economist
    ```

2. Create a new conda environment (named "ai-economist" below - replace with anything else) and activate it

    ```pyfunctiontypecomment
    conda create --name ai-economist python=3.6
    conda activate ai-economist
    ```

3. Either

    a) Edit the PYTHONPATH to include the ai-economist directory
    ```
    export PYTHONPATH=<local path to ai-economist>:$PYTHONPATH
    ```

    OR

    b) Install as an editable Python package
    ```pyfunctiontypecomment
    cd ai-economist
    pip install -e .
    ```

Useful tip: for quick access, add the following to your ~/.bashrc or ~/.bash_profile:

```pyfunctiontypecomment
alias aiecon="conda activate ai-economist; cd <local path to ai-economist>"
```

You can then simply run `aiecon` once to activate the conda environment.

### Testing your Install

To test your installation, try running:

```
conda activate ai-economist
python -c "import ai_economist"
```
## Structure of the Code

The simulation is located in the `ai_economist/foundation` folder.

The code repository is organized into the following components:

| Component | Description |
| --- | --- |
| [base](https://www.github.com/salesforce/ai-economist/blob/master/ai_economist/foundation/base) | Contains base classes to can be extended to define Agents, Components and Scenarios. |
| [agents](https://www.github.com/salesforce/ai-economist/blob/master/ai_economist/foundation/agents) | Agents represent economic actors in the environment. Currently, we have mobile Agents (representing workers) and a social planner (representing a government). |
| [entities](https://www.github.com/salesforce/ai-economist/blob/master/ai_economist/foundation/entities) | Endogenous and exogenous components of the environment. Endogenous entities include labor, while exogenous entity includes landmarks (such as Water and Grass) and collectible Resources (such as Wood and Stone). |
| [components](https://www.github.com/salesforce/ai-economist/blob/master/ai_economist/foundation/components) | Components are used to add some particular dynamics to an environment. They also add action spaces that define how Agents can interact with the environment via the Component. |
| [scenarios](https://www.github.com/salesforce/ai-economist/blob/master/ai_economist/foundation/scenarios) | Scenarios compose Components to define the dynamics of the world. It also computes rewards and exposes state for visualization. |


## Simulation Notebooks

- [economic_simulation_basic.ipynb](https://github.com/swarnabha13/ai-economist/blob/master/economic_simulation_basic.ipynb)
- [AI_Economist_RL.ipynb](https://github.com/swarnabha13/ai-economist/blob/master/AI_Economist_RL.ipynb)
- [SARSA_Test.ipynb](https://github.com/swarnabha13/ai-economist/blob/master/SARSA_Test.ipynb)
