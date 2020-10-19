from typing import List

import numpy as np
import pandas as pd
from bokeh.embed import file_html  # type: ignore
from bokeh.models import Panel, Tabs  # type: ignore
from bokeh.plotting import figure  # type: ignore
from bokeh.resources import CDN  # type: ignore
from matplotlib import cm

from .task import Task


def make_colors(N):
    colors = []
    cmap = cm.get_cmap("RdYlGn")
    x = cmap(np.linspace(1.0, 0.0, num=N), bytes=True)
    for row in x:
        colors.append(f"rgb({row[0]}, {row[1]}, {row[2]})")
    return colors


class TaskGrid:
    """This class represents a set of tasks.

    Attributes:
        tasks: A list of benchmark tasks.
    """

    def __init__(self, tasks: List[Task]):
        self.tasks = {}
        for task in tasks:
            self.tasks[task.name] = task

    def to_html(self, path_to_html):
        """Export to html file.

        Args:
            path_to_html: The path to write the HTML.
        """
        html = file_html(self.to_bokeh(), CDN)
        with open(path_to_html, "wt") as fout:
            fout.write(html)

    def to_bokeh(self):
        """Export to bokeh Figure."""
        tabs = []
        tabs.append(Panel(child=self._bokeh_overview(), title="overview"))
        for task in self.tasks.values():
            tabs.append(Panel(child=task.to_bokeh(), title=task.name))
        return Tabs(tabs=tabs)

    def _bokeh_overview(self):
        x_range = list(sorted(set(self.tasks.keys())))
        y_range = set()
        for task in self.tasks.values():
            y_range.update(task.results.keys())
        y_range = list(sorted(y_range, reverse=True))
        fig = figure(
            plot_width=650,
            plot_height=650,
            toolbar_location=None,
            x_range=[f"{x} ({self.tasks[x].default_metric.name})" for x in x_range],
            y_range=y_range,
        )

        x, y, colors = [], [], []
        for (i, task_name) in enumerate(x_range):
            task = self.tasks[task_name]

            gradient = make_colors(len(task.rank()))
            model_to_color = {
                model: color for model, color in zip(task.rank(), gradient)
            }

            for (j, model_name) in enumerate(y_range):
                if model_name not in task.results:
                    continue
                x.append(i + 0.5)
                y.append(j + 0.5)
                colors.append(model_to_color[model_name])

        fig.rect(x=x, y=y, width=0.9, height=0.9, color=colors)
        return fig

    def to_df(self) -> pd.DataFrame:
        dfs = []
        for task in self.tasks.values():
            df = task.to_df()
            df.insert(0, "task", task.name)
            dfs.append(df)
        return pd.concat(dfs, axis=0)

    def to_csv(self, path_to_csv):
        """Export to CSV.

        Args:
            path_to_csv: The path to write the csv.
        """
        self.to_df().to_csv(path_to_csv, index=False)
