import unittest

from metricx import Metric, Selector, Task


class TestSelector(unittest.TestCase):
    def test_repr(self):
        task = Task(
            name="hello-world",
            metrics=[
                Metric(name="score", is_higher_better=True),
                Metric(name="fit-time", is_higher_better=False),
            ],
        )
        selector = Selector(task)

        task.report("model-1", {"score": 1.0, "fit-time": 1.0})
        assert selector.propose() == "model-1"

        for _ in range(10):
            model = selector.propose()
            task.report(model, {"score": 1.0, "fit-time": 1.0})
