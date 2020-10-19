import os
import tempfile
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
        task.report("model-1", {"score": 1.1, "fit-time": 1.0})
        grid = TaskGrid([task])
        self.assertEqual(len(grid.to_df()), 2)
        self.assertTrue(grid.to_bokeh())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "index.html")
            self.assertTrue(not os.path.exists(path))
            grid.to_html(path)
            self.assertTrue(os.path.exists(path))
