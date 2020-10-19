# MetricX

[![Build Status](https://github.com/k15z/MetricX/workflows/Build%20Main/badge.svg)](https://github.com/k15z/MetricX/actions)
[![Documentation](https://github.com/k15z/MetricX/workflows/Documentation/badge.svg)](https://k15z.github.io/MetricX)
[![Code Coverage](https://codecov.io/gh/k15z/MetricX/branch/main/graph/badge.svg)](https://codecov.io/gh/k15z/MetricX)

A library for managing, exploring, and analyzing benchmark data. Given a set of tasks
and a set of models which can be evaluated on the tasks, `MetricX` provides tools 
for keeping track of results, exporting the benchmark results, and generating 
interactive visualizations of the data.

---

## Quick Start

### Install MetricX
You can install the latest stable release

```bash
pip install metricx
```
Or you can install the development head

```bash
pip install git+https://github.com/k15z/MetricX.git
```

### Define your task
First, you'll want to define your task(s). Every task has a name and a
set of metrics.

```python
from metricx import Metric, Task

task = Task(
    name="mnist-digits",
    metrics=[
        Metric(name="accuracy", is_higher_better=True),
        Metric(name="fit-time", is_higher_better=False),
        Metric(name="predict-time", is_higher_better=False),
    ],
)
```

### Report your results
Then, you can report your results by providing (1) the name of the model
being evaluated and (2) a results dictionary containing a value for each of
the metrics specified earlier.

```python
task.report("convnet", {
  "accuracy": 1.0, 
  "fit-time": 100.0,
  "predict-time": 3.0,
})
```

```python
task.report("logistic-regression", {
  "accuracy": 0.6, 
  "fit-time": 10.0,
  "predict-time": 1.0,
})
```

### Generate plots and rankings
The :class:`~metricx.Task` object provides numerous functionality from 
generating plots to ranking models.

```python
task.to_bokeh() # generate a Bokeh plot
task.to_figure() # generate a matplotlib Figure
task.rank() # return a ranking of models
```

### Combine multiple tasks
If you have multiple tasks, as is typical in a benchmarking scenario, you 
can use the :class:`~metricx.TaskGrid` to wrap them together and generated 
combined visualizations and more.

```python
from metricx import TaskGrid
from bokeh.io import output_notebook, show

output_notebook()
grid = TaskGrid([task])
show(grid.to_bokeh())
```
