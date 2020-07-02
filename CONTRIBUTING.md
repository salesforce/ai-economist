# Contributing to AI-Economist

Thanks for considering contributing to AI-Economist!

A typical workflow is to start a discussion about bugs or new features in a Github Issue, and work together with us to choose a strategy for a fix or enhancement. We recommend that all code changes are submitted via a Pull Request so we can go over the review process and apply the necessary changes. Please keep the PRs small so they are easy to understand and quick to review.

# What to Contribute?

Some of the things we are particularly interested in, include:

- New Components
- New Scenarios
- Learning algorithms
- Policy models
- Documentation, examples, and tutorials

However, we are always interested in hearing from you and your ideas!

# Code Formatting and Checks

Code is formatted using [isort](https://github.com/timothycrosley/isort) and [black](https://black.readthedocs.io/en/stable/). We also run a flake8 and pylint check on the code, with some exceptions, such as ignoring bad-continuation (C0330) errors. Before your code can be reviewed, please make sure it passes these basic checks.
We have also included a [shell script](https://github.com/salesforce/ai-economist/blob/master/format_and_lint.sh) in the home directory that you can use to format as well as check the code.

If you're adding new functionality that merits a quick unit test, please try to include that under the `tests` folder :).
