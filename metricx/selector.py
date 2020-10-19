from collections import Counter
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .metric import Metric
from .task import Task


def random_policy(task: Task, metric: Optional[Union[str, Metric]] = None) -> str:
    """Select a random model."""
    return np.random.choice(list(task.results.keys()))


def stderr_policy(task: Task, metric: Optional[Union[str, Metric]] = None) -> str:
    """Select the model with the largest variance."""
    mu_var_n = task.model_to_mu_var_n(metric)
    stderr_model = [
        (np.sqrt(var / n), model) for model, (mu, var, n) in mu_var_n.items()
    ]
    _, model = list(sorted(stderr_model))[-1]
    return model


def power_policy(task: Task, metric: Optional[Union[str, Metric]] = None) -> str:
    """Select the model with the most samples needed to achieve power."""
    model_to_samples: Counter = Counter()
    for modelA in task.results.keys():
        model_to_samples[modelA] = 0
        for modelB in task.results.keys():
            if modelA == modelB:
                continue
            nA, nB = task.samples_to_achieve_power(modelA, modelB, metric)
            model_to_samples[modelA] += nA
            model_to_samples[modelB] += nB
    model, _ = model_to_samples.most_common(1)[0]
    return model


class Selector:
    """This class implements methods for selecting models to run.

    Given a target metric, the Selector class provides methods for selecting
    models to run. The default policy is to:

        1. Obtain `min_samples=3` for each model.
        2. Sample from `policies`:

            - (p=0.3) Sample a random model.
            - (p=0.35) Select the model with the largest standard errror.
            - (p=0.35) Select the model with the largest number of samples
              needed to achieve power.

    Attributes:
        task: The target task.
        policies: A list of tuples containing a policy and the probability of
            selecting that policy.
        min_samples: The minimum number of samples required for each model
            before the randomized policies are applied.
    """

    DEFAULT_POLICY = [
        (random_policy, 0.3),
        (stderr_policy, 0.35),
        (power_policy, 0.35),
    ]

    def __init__(
        self,
        task: Task,
        policies: List[Tuple[Callable, float]] = [],
        min_samples: int = 3,
    ):
        self.task = task
        self.min_samples = min_samples
        self.policies = policies if policies else self.DEFAULT_POLICY

    def propose(self, metric: Optional[Union[str, Metric]] = None) -> str:
        """This selects a model to execute.

        Args:
            metric: The target metric to optimize.

        Returns:
            The model to execute.
        """
        for model, results in self.task.results.items():
            if len(results) < self.min_samples:
                return model
        policies, probabilities = zip(*self.policies)
        policy = np.random.choice(policies, p=probabilities)  # type: ignore
        return policy(self.task, metric)
