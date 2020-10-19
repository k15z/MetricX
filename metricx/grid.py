from typing import List

from bokeh.models import Panel, Tabs

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

    def visualize(self):
        tabs = []
        for task in self.tasks.values():
            tabs.append(Panel(child=task.to_bokeh(), title=task.name))
        return Tabs(tabs=tabs)
