import unittest
from random import random

from metricx import Metric, Task


class TestTask(unittest.TestCase):
    def test_rank_best(self):
        task = Task(
            name="hello-world",
            metrics=[
                Metric(name="score", is_higher_better=True),
                Metric(name="fit-time", is_higher_better=False),
            ],
        )
        task.report("model-1", {"score": 1.0, "fit-time": 1.0})
        task.report("model-2", {"score": 0.0, "fit-time": 0.0})
        self.assertEqual(task.rank("score"), ["model-1", "model-2"])
        self.assertEqual(task.rank("fit-time"), ["model-2", "model-1"])
        self.assertEqual(task.best("score"), "model-1")
        with self.assertRaises(ValueError):
            task.samples_to_achieve_power("model-1", "model-2")

        for _ in range(10):
            task.report("model-1", {"score": random(), "fit-time": 1.0})
            task.report("model-2", {"score": random(), "fit-time": 0.0})
        self.assertTrue(task.samples_to_achieve_power("model-1", "model-2"))

    def test_export(self):
        task = Task(
            name="hello-world",
            metrics=[
                Metric(name="score", is_higher_better=True),
                Metric(name="fit-time", is_higher_better=False),
            ],
        )
        self.assertEqual(len(task.to_df()), 0)

        task.report("model-1", {"score": 1.0, "fit-time": 1.0})
        task.report("model-2", {"score": 0.0, "fit-time": 0.0})
        task.report("model-2", {"score": 0.0, "fit-time": 0.1})
        self.assertEqual(len(task.to_df()), 3)

        self.assertTrue(task.to_figure())
        self.assertTrue(task.to_bokeh())
