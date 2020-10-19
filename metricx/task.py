from itertools import cycle
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm  # type: ignore

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
    def __init__(
        self,
        name: str,
        metrics: List[Metric],
    ):
        self.name = name
        self.primary_metric = metrics[0]
        self.metrics: Dict[str, Metric] = {}
        self.results: Dict[str, List[Dict[str, float]]] = {}

        for metric in metrics:
            assert metric.name not in self.metrics
            self.metrics[metric.name] = metric

    def report(self, model: str, result: Dict[Union[str, Metric], float]):
        if model not in self.results:
            self.results[model] = []

        result_: Dict[str, float] = {}
        for metric in self.metrics.values():
            if metric in result:
                result_[metric.name] = result[metric]
            else:
                result_[metric.name] = result[metric.name]
        self.results[model].append(result_)

    def rank(self, metric: Optional[Union[str, Metric]] = None) -> List[str]:
        """Return a list of models from best to worst."""
        metric = self._get_metric(metric)

        scores = []
        for model, (mu, var, _) in self._model_to_mu_var_n(metric).items():
            if metric.is_higher_better:
                mu = -mu
            scores.append((mu, var, model))
        scores = sorted(scores)

        _, _, models = zip(*scores)
        return list(models)

    def best(self, metric: Union[str, Metric]) -> str:
        """Return the best model."""
        return self.rank(metric)[0]

    def propose(self, metric: Optional[Union[str, Metric]] = None) -> str:
        """Propose a model to evaluate on this task."""
        return np.random.choice(list(self.results.keys()))

    def _get_metric(self, metric: Optional[Union[str, Metric]]) -> Metric:
        if metric is None:
            return self.primary_metric
        elif isinstance(metric, str):
            return self.metrics[metric]
        return metric

    def _model_to_mu_var_n(self, metric: Optional[Union[str, Metric]]):
        """Compute mean, variance, and count."""
        metric = self._get_metric(metric)
        model_to_mu_var_n = {}
        for model, results in self.results.items():
            values = np.array([result[metric.name] for result in results])
            mu, var = np.mean(values), np.var(values) + 1e-5  # type: ignore
            if len(values) <= 1:
                var = 0.0
            model_to_mu_var_n[model] = (mu, var, len(values))
        return model_to_mu_var_n

    def to_csv(self, path_to_csv):
        self.to_df.to_csv(path_to_csv, index=False)

    def to_df(self) -> pd.DataFrame:
        rows = []
        for model, results in self.results.items():
            for result in results:
                obj: Dict[str, Any] = {}
                obj["model"] = model
                obj.update(result)
                rows.append(obj)
        return pd.DataFrame(rows)

    def to_figure(self) -> plt.Figure:
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
                    sigma = 1e-5
                    sigma = grp[metric.name].std()
                    x = np.linspace(xmin, xmax, 1000)
                    y = norm.pdf(x, loc=mu, scale=sigma)
                    axs[i].plot(x, y, label=f"{model}", color=next(colors))
            axs[i].set_xlim(xmin, xmax)
        axs[0].set_title(self.name)
        axs[0].legend()
        return fig
