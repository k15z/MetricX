import unittest

from metricx import Metric, Task, TaskGrid


class TestTaskGrid(unittest.TestCase):
    def test_grid(self):
        task = Task(
            name="hello-world",
            metrics=[
                Metric(name="score", is_higher_better=True),
                Metric(name="fit-time", is_higher_better=False),
            ],
        )
        task.report("model-1", {"score": 1.0, "fit-time": 1.0})
        grid = TaskGrid([task])
        assert len(grid.to_df()) == 1
        assert grid.to_bokeh()
