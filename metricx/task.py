from datetime import datetime
from itertools import cycle
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.layouts import column  # type: ignore
from bokeh.models import Range1d  # type: ignore
from bokeh.plotting import figure  # type: ignore
from scipy.stats import norm  # type: ignore
from statsmodels.stats.power import tt_ind_solve_power  # type: ignore

from metricx.metric import Metric

_colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _is_unique(s):
    a = s.to_numpy()
    return (a[0] == a).all()


class Task:
    """This class represents a task which is used to evaluate models.

    Attributes:
        name: The name of the task.
        metrics: A list of Metrics which are produced by models which are
            executed on the task. The first metric in this list is considered
            the default metric.
    """

    def __init__(
        self,
        name: str,
        metrics: List[Metric],
    ):
        self.name = name
        self.metrics: Dict[str, Metric] = {}
        self.results: Dict[str, List[Dict[str, float]]] = {}

        for metric in metrics:
            assert metric.name not in self.metrics
            self.metrics[metric.name] = metric
        self.default_metric = metrics[0]

    def report(self, model: str, result: Dict[Union[str, Metric], float]):
        """Report a result for the specified model.

        Args:
            model: A string identifying the model.
            result: A dictionary mapping each metric to its value.

        Raises:
            KeyError: If the result dictionary is missing metrics.
        """
        if model not in self.results:
            self.results[model] = []

        result_: Dict[str, Any] = {}
        result_["timestamp"] = datetime.now()
        for metric in self.metrics.values():
            if metric in result:
                result_[metric.name] = result[metric]
            else:
                result_[metric.name] = result[metric.name]
        self.results[model].append(result_)

    def rank(self, metric: Optional[Union[str, Metric]] = None) -> List[str]:
        """Rank the models.

        Args:
            metric: The target metric to sort by.

        Returns:
            A list of models, sorted from best to worst.
        """
        metric = self.get_metric(metric)

        scores = []
        for model, (mu, var, _) in self.model_to_mu_var_n(metric).items():
            if metric.is_higher_better:
                mu = -mu
            scores.append((mu, var, model))
        scores = sorted(scores)

        _, _, models = zip(*scores)
        return list(models)

    def best(self, metric: Union[str, Metric]) -> str:
        """Get the best model.

        Args:
            metric: The target metric to sort by.

        Returns:
            The best model on this task.
        """
        return self.rank(metric)[0]

    def samples_to_achieve_power(
        self, modelA: str, modelB: str, metric: Optional[Union[str, Metric]] = None
    ):
        """Number of samples needed to achieve power.

        This method estimates the number of samples needed - for each
        model - to achieve 50% statistical power. This corresponds to the
        probability of detecting a statistically significant effect
        (p-value=0.1) if there is one.
        """
        model_to_mu_var_n = self.model_to_mu_var_n(metric)
        muA, varA, nA = model_to_mu_var_n[modelA]
        muB, varB, nB = model_to_mu_var_n[modelB]
        if nA < 3 or nB < 3:
            raise ValueError("Not enough samples to estimate the power.")

        pct_change = abs(0.1 * (muA + muB) / 2.0)
        std_dev = np.sqrt(varA + varB)
        effect_size = pct_change / std_dev

        nA_needed = tt_ind_solve_power(
            effect_size=effect_size, nobs1=None, alpha=0.1, power=0.5, ratio=nB / nA
        )
        nB_needed = nA_needed * nB / nA
        return max(0, nA_needed - nA), max(0, nB_needed - nB)

    def get_metric(self, metric: Optional[Union[str, Metric]]) -> Metric:
        if metric is None:
            return self.default_metric
        elif isinstance(metric, str):
            return self.metrics[metric]
        return metric

    def model_to_mu_var_n(self, metric: Optional[Union[str, Metric]]):
        """Compute mean, variance, and count."""
        metric = self.get_metric(metric)
        model_to_mu_var_n = {}
        for model, results in self.results.items():
            values = np.array([result[metric.name] for result in results])
            mu, var = np.mean(values), np.var(values) + 1e-5  # type: ignore
            if len(values) <= 1:
                var = 0.0
            model_to_mu_var_n[model] = (mu, var, len(values))
        return model_to_mu_var_n

    def to_csv(self, path_to_csv):
        """Export to CSV.

        Args:
            path_to_csv: The path to write the csv.
        """
        self.to_df().to_csv(path_to_csv, index=False)

    def to_df(self) -> pd.DataFrame:
        """Export to DataFrame.

        Returns:
            A DataFrame where each row corresponds to a single run.
        """
        rows = []
        for model, results in self.results.items():
            for result in results:
                obj: Dict[str, Any] = {}
                obj["model"] = model
                obj.update(result)
                rows.append(obj)
        return pd.DataFrame(rows)

    def to_figure(self) -> plt.Figure:
        """Export to Figure.

        Returns:
            A Figure where each subplot shows a metric.
        """
        fig, axs = plt.subplots(len(self.metrics), figsize=(10, 2 * len(self.metrics)))
        if not isinstance(axs, np.ndarray):
            axs = [axs]  # type: ignore

        df = self.to_df()
        for i, metric in enumerate(self.metrics.values()):
            colors = cycle(_colors)
            axs[i].set_ylabel(metric.name)
            xmin = df[metric.name].min() - df[metric.name].std() * 3
            xmax = df[metric.name].max() + df[metric.name].std() * 3
            for model, grp in df.groupby("model"):
                mu = grp[metric.name].mean()
                if _is_unique(grp[metric.name]):
                    axs[i].axvline(  # type: ignore
                        mu, 0.0, 1.0, label=f"{model}", color=next(colors)
                    )
                else:
                    sigma = grp[metric.name].std()
                    x = np.linspace(xmin, xmax, 1000)
                    y = norm.pdf(x, loc=mu, scale=sigma)
                    axs[i].plot(x, y, label=f"{model}", color=next(colors))
            axs[i].set_xlim(xmin, xmax)
        axs[0].set_title(self.name)
        axs[0].legend()
        return fig

    def to_bokeh(self):
        """Export to bokeh Figure."""
        figures = []

        df = self.to_df()
        for i, metric in enumerate(self.metrics.values()):
            fig = figure(
                plot_width=650,
                plot_height=200,
                title=metric.name,
                toolbar_location=None,
            )

            colors = cycle(_colors)
            ymax = 0.0
            xmin = df[metric.name].min() - df[metric.name].std() * 3
            xmax = df[metric.name].max() + df[metric.name].std() * 3
            for model, grp in df.groupby("model"):
                mu = grp[metric.name].mean()
                if _is_unique(grp[metric.name]):
                    sigma = grp[metric.name].std()
                    x = [mu, mu]
                    y = [0, 100.0]
                    fig.line(x, y, line_color=next(colors), legend_label=model)

                else:
                    sigma = grp[metric.name].std()
                    x = np.linspace(xmin, xmax, 1000)
                    y = norm.pdf(x, loc=mu, scale=sigma)
                    fig.line(x, y, line_color=next(colors), legend_label=model)
                    ymax = max(ymax, np.max(y))

            fig.x_range = Range1d(xmin, xmax)
            fig.y_range = Range1d(0.0, ymax * 1.1)
            figures.append(fig)

            fig.legend.visible = i == 0

        return column(*figures)
