import unittest

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
        assert task.rank("score") == ["model-1", "model-2"]
        assert task.rank("fit-time") == ["model-2", "model-1"]
        assert task.best("score") == "model-1"

    def test_export(self):
        task = Task(
            name="hello-world",
            metrics=[
                Metric(name="score", is_higher_better=True),
                Metric(name="fit-time", is_higher_better=False),
            ],
        )
        assert len(task.to_df()) == 0

        task.report("model-1", {"score": 1.0, "fit-time": 1.0})
        task.report("model-2", {"score": 0.0, "fit-time": 0.0})
        task.report("model-2", {"score": 0.0, "fit-time": 0.1})
        assert len(task.to_df()) == 3

        assert task.to_figure()
        assert task.to_bokeh()
