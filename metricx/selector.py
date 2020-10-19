from random import choice
from typing import Optional, Union

from .metric import Metric
from .task import Task


class Selector:
    """This class implements methods for selecting models to run.

    Given a target metric, the Selector class provides methods for selecting
    models to run. Examples of criteria that could be used include:

        1. Choosing the model with the fewest samples.
        2. Choosing the model with the highest variance.
        3. Choosing the model such that a new sample would produce the greatest
           change in the expected ranking.

    Attributes:
        task: The target task.
    """

    def __init__(self, task: Task):
        self.task = task

    def propose(self, metric: Optional[Union[str, Metric]] = None) -> str:
        """This selects a model to execute.

        Args:
            metric: The target metric to optimize.

        Returns:
            The model to execute.
        """
        for model, results in self.task.results.items():
            if len(results) < 3:
                return model
        return choice(list(self.task.results.keys()))
