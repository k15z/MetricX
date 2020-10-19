import unittest

from metricx import Metric


class TestMetric(unittest.TestCase):
    def test_repr(self):
        metric = Metric(
            name="auroc", description="Area under ROC curve.", is_higher_better=True
        )
        assert repr(metric) == "Metric('auroc', 'Area under ROC curve.', True)"

    def test_equality(self):
        metric1 = Metric(
            name="auroc", description="Area under ROC curve.", is_higher_better=True
        )
        metric2 = Metric(
            name="auroc", description="Area under ROC curve.", is_higher_better=True
        )
        metric3 = Metric(
            name="recall@10",
            description="Recall at top ten.",
            is_higher_better=True,
        )
        assert metric1 == metric2
        assert metric1 != metric3
