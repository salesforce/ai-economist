# Foundation: An Economic Simulation Framework

This repo contains an implementation of Foundation, a framework for flexible, modular, and composable environments that **model socio-economic behaviors and dynamics in a society with both agents and governments**.

Foundation provides a [Gym](https://gym.openai.com/)-style API:

- `reset`: resets the environment's state and returns the observation.
- `step`: advances the environment by one timestep, and returns the tuple *(observation, reward, done, info)*.

This simulation can be used in conjunction with reinforcement learning to learn optimal economic policies, as detailed in the following papers:

**[The AI Economist: Improving Equality and Productivity with AI-Driven Tax Policies](https://arxiv.org/abs/2004.13332)**,
*Stephan Zheng, Alexander Trott, Sunil Srinivasa, Nikhil Naik, Melvin Gruesbeck, David C. Parkes, Richard Socher.*

**[The AI Economist: Optimal Economic Policy Design via Two-level Deep Reinforcement Learning](https://arxiv.org/abs/2108.02755)**
*Stephan Zheng, Alexander Trott, Sunil Srinivasa, David C. Parkes, Richard Socher.*

**[Building a Foundation for Data-Driven, Interpretable, and Robust Policy Design using the AI Economist](https://arxiv.org/abs/2108.02904)**
*Alexander Trott, Sunil Srinivasa, Douwe van der Wal, Sebastien Haneuse, Stephan Zheng.*

If you use this code in your research, please cite us using this BibTeX entry:

```
@misc{2004.13332,
 Author = {Stephan Zheng, Alexander Trott, Sunil Srinivasa, Nikhil Naik, Melvin Gruesbeck, David C. Parkes, Richard Socher},
 Title = {The AI Economist: Improving Equality and Productivity with AI-Driven Tax Policies},
 Year = {2020},
 Eprint = {arXiv:2004.13332},
}
```

For more information and context, check out:

- [The AI Economist website](https://www.einstein.ai/the-ai-economist)
- [Blog: The AI Economist: Improving Equality and Productivity with AI-Driven Tax Policies](https://blog.einstein.ai/the-ai-economist/)
- [Blog: The AI Economist moonshot](https://blog.einstein.ai/the-ai-economist-moonshot/)
- [Blog: The AI Economist web demo of the COVID-19 case study](https://blog.einstein.ai/ai-economist-covid-case-study-ethics/)
- [Web demo: The AI Economist ethical review of AI policy design and COVID-19 case study](https://einstein.ai/the-ai-economist/ai-policy-foundation-and-covid-case-study)

## Simulation Cards: Ethics Review and Intended Use

Please see our [Simulation Card](https://github.com/salesforce/ai-economist/blob/master/Simulation_Card_Foundation_Economic_Simulation_Framework.pdf) for a review of the intended use and ethical review of our framework.

Please see our [COVID-19 Simulation Card](https://github.com/salesforce/ai-economist/blob/master/COVID-19_Simulation-Card.pdf) for a review of the ethical aspects of the pandemic simulation (and as fitted for COVID-19).

## Join us on Slack

If you're interested in extending this framework, discussing machine learning for economics, and collaborating on research project:

- join our Slack channel [aieconomist.slack.com](https://aieconomist.slack.com) using this [invite link](https://join.slack.com/t/aieconomist/shared_invite/zt-g71ajic7-XaMygwNIup~CCzaR1T0wgA), or
- email us @ ai.economist@salesforce.com.

## Installation Instructions

To get started, you'll need to have Python 3.7+ installed.

### Using pip

Simply use the Python package manager:

```python
pip install ai-economist
```

### Installing from Source

1. Clone this repository to your local machine:

  ```
   git clone www.github.com/salesforce/ai-economist
   ```

2. Create a new conda environment (named "ai-economist" below - replace with anything else) and activate it

  ```pyfunctiontypecomment
   conda create --name ai-economist python=3.7 --yes
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

## Getting Started

### Tutorials

To familiarize yourself with Foundation, check out the tutorials in the `tutorials` folder. You can run these notebooks interactively in your browser on Google Colab.

- [tutorials/economic_simulation_basic.ipynb](https://www.github.com/salesforce/ai-economist/blob/master/tutorials/economic_simulation_basic.ipynb) ([Try this on Colab](http://colab.research.google.com/github/salesforce/ai-economist/blob/master/tutorials/economic_simulation_basic.ipynb)!): Shows how to interact with and visualize the simulation.
- [tutorials/economic_simulation_advanced.ipynb](https://www.github.com/salesforce/ai-economist/blob/master/tutorials/economic_simulation_advanced.ipynb) ([Try this on Colab](http://colab.research.google.com/github/salesforce/ai-economist/blob/master/tutorials/economic_simulation_advanced.ipynb)!): Explains how Foundation is built up using composable and flexible building blocks.
- [tutorials/covid19_and_economic_simulation.ipynb](https://www.github.com/salesforce/ai-economist/blob/master/tutorials/covid19_and_economic_simulation.ipynb) ([Try this on Colab](http://colab.research.google.com/github/salesforce/ai-economist/blob/master/tutorials/covid19_and_economic_simulation.ipynb)!): A tutorial on how to work specifically with the COVID-19 pandemic and economy simulation.

To run these notebooks *locally*, you need [Jupyter](https://jupyter.org). See [https://jupyter.readthedocs.io/en/latest/install.html](https://jupyter.readthedocs.io/en/latest/install.html) for installation instructions and [(https://jupyter-notebook.readthedocs.io/en/stable/](https://jupyter-notebook.readthedocs.io/en/stable/) for examples of how to work with Jupyter.

## Structure of the Code

- The simulation is located in the `ai_economist/foundation` folder.

The code repository is organized into the following components:

| Component | Description |
| --- | --- |
| [base](https://www.github.com/salesforce/ai-economist/blob/master/ai_economist/foundation/base) | Contains base classes to can be extended to define Agents, Components and Scenarios. |
| [agents](https://www.github.com/salesforce/ai-economist/blob/master/ai_economist/foundation/agents) | Agents represent economic actors in the environment. Currently, we have mobile Agents (representing workers) and a social planner (representing a government). |
| [entities](https://www.github.com/salesforce/ai-economist/blob/master/ai_economist/foundation/entities) | Endogenous and exogenous components of the environment. Endogenous entities include labor, while exogenous entity includes landmarks (such as Water and Grass) and collectible Resources (such as Wood and Stone). |
| [components](https://www.github.com/salesforce/ai-economist/blob/master/ai_economist/foundation/components) | Components are used to add some particular dynamics to an environment. They also add action spaces that define how Agents can interact with the environment via the Component. |
| [scenarios](https://www.github.com/salesforce/ai-economist/blob/master/ai_economist/foundation/scenarios) | Scenarios compose Components to define the dynamics of the world. It also computes rewards and exposes states for visualization. |

- The datasets (including the real-world data on COVID-19) are located in the `ai_economist/datasets` folder.

## Releases and Contributing

- Please let us know if you encounter any bugs by filing a Github issue.
- We appreciate all your contributions. If you plan to contribute new Components, Scenarios Entities, or anything else, please see our [contribution guidelines](https://www.github.com/salesforce/ai-economist/blob/master/CONTRIBUTING.md).

## Changelog

Current version: v1.2

For the complete release history, see [CHANGELOG.md](https://www.github.com/salesforce/ai-economist/blob/master/CHANGELOG.md).

## License

Foundation and the AI Economist are released under the [BSD-3 License](LICENSE.txt).

