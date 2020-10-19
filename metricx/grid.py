from typing import List

import pandas as pd
from bokeh.models import Panel, Tabs  # type: ignore

from .task import Task


def _is_unique(s):
    a = s.to_numpy()
    return (a[0] == a).all()


class TaskGrid:
    """This class represents a set of tasks.

    Attributes:
        tasks: A list of benchmark tasks.
    """

    def __init__(self, tasks: List[Task]):
        self.tasks = {}
        for task in tasks:
            self.tasks[task.name] = task

    def to_bokeh(self):
        tabs = []
        for task in self.tasks.values():
            tabs.append(Panel(child=task.to_bokeh(), title=task.name))
        return Tabs(tabs=tabs)

    def to_df(self) -> pd.DataFrame:
        dfs = []
        for task in self.tasks.values():
            dfs.append(task.to_df())
        return pd.concat(dfs, axis=0)

    def to_csv(self, path_to_csv):
        """Export to CSV.

        Args:
            path_to_csv: The path to write the csv.
        """
        self.to_df().to_csv(path_to_csv, index=False)
