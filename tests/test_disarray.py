import disarray
import pandas as pd
import unittest

from disarray.metrics import __all_metrics__


class TestDisarray(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_binary = pd.DataFrame([[50, 10], [10, 30]])
        cls.classes = ['setosa', 'versicolor', 'virginica']
        cls.df_multi = pd.DataFrame([[13, 0, 0], [0, 10, 6], [0, 0, 9]], index=cls.classes, columns=cls.classes)

    def test_all_metrics(self):
        with self.assertRaises(AttributeError):
            getattr(self.df_binary.da, 'unused-metric')

        detected_metrics = []
        for metric in __all_metrics__:
            if isinstance(getattr(self.df_binary.da, metric), pd.Series):
                detected_metrics.append(metric)
        self.assertCountEqual(__all_metrics__, detected_metrics)

    def test_accuracy(self):
        self.assertAlmostEqual(self.df_binary.da.accuracy.loc[0], 0.80, 2)
        self.assertAlmostEqual(self.df_binary.da.micro_accuracy, 0.80, 2)
        self.assertAlmostEqual(self.df_multi.da.accuracy.loc['setosa'], 1.0, 2)
        self.assertAlmostEqual(self.df_multi.da.micro_accuracy, 0.89, 2)

    def test_f1(self):
        self.assertAlmostEqual(self.df_binary.da.f1.loc[0], 0.83, 2)
        self.assertAlmostEqual(self.df_binary.da.micro_f1, 0.80, 2)
        self.assertAlmostEqual(self.df_multi.da.f1.loc['setosa'], 1.0, 2)
        self.assertAlmostEqual(self.df_multi.da.micro_f1, 0.84, 2)

    def test_false_discovery_rate(self):
        self.assertAlmostEqual(self.df_binary.da.false_discovery_rate.loc[0], 0.17, 2)
        self.assertAlmostEqual(self.df_binary.da.micro_false_discovery_rate, 0.20, 2)
        self.assertAlmostEqual(self.df_multi.da.false_discovery_rate.loc['setosa'], 0.0, 2)
        self.assertAlmostEqual(self.df_multi.da.micro_false_discovery_rate, 0.16, 2)
        self.assertAlmostEqual(self.df_multi.da.micro_false_discovery_rate,
                               1 - self.df_multi.da.micro_precision, 2)

    def test_false_negative_rate(self):
        self.assertAlmostEqual(self.df_binary.da.false_negative_rate.loc[0], 0.166, 2)
        self.assertAlmostEqual(self.df_binary.da.micro_false_negative_rate, 0.20, 2)
        self.assertAlmostEqual(self.df_multi.da.false_negative_rate.loc['setosa'], 0.0, 2)
        self.assertAlmostEqual(self.df_multi.da.micro_false_negative_rate, 0.16, 2)
        self.assertAlmostEqual(self.df_multi.da.micro_false_negative_rate,
                               1 - self.df_multi.da.micro_true_positive_rate, 2)

    def test_false_positive_rate(self):
        self.assertAlmostEqual(self.df_binary.da.false_positive_rate.loc[0], 0.25, 2)
        self.assertAlmostEqual(self.df_binary.da.micro_false_positive_rate, 0.20, 2)
        self.assertAlmostEqual(self.df_multi.da.false_positive_rate.loc['setosa'], 0.0, 2)
        self.assertAlmostEqual(self.df_multi.da.micro_false_positive_rate, 0.08, 2)
        self.assertAlmostEqual(self.df_multi.da.micro_false_positive_rate,
                               1 - self.df_multi.da.micro_true_negative_rate, 2)

    def test_negative_predictive_value(self):
        self.assertAlmostEqual(self.df_binary.da.negative_predictive_value.loc[0], 0.75, 2)
        self.assertAlmostEqual(self.df_binary.da.micro_negative_predictive_value, 0.80, 2)
        self.assertAlmostEqual(self.df_multi.da.negative_predictive_value.loc['setosa'], 1.0, 2)
        self.assertAlmostEqual(self.df_multi.da.micro_negative_predictive_value, 0.92, 2)

    def test_positive_predictive_value(self):
        self.assertAlmostEqual(self.df_binary.da.positive_predictive_value.loc[0], 0.83, 2)
        self.assertAlmostEqual(self.df_binary.da.micro_positive_predictive_value, 0.80, 2)
        self.assertAlmostEqual(self.df_multi.da.positive_predictive_value.loc['setosa'], 1.0, 2)
        self.assertAlmostEqual(self.df_multi.da.micro_positive_predictive_value, 0.84, 2)
        self.assertAlmostEqual(self.df_multi.da.micro_positive_predictive_value,
                               1 - self.df_multi.da.micro_false_discovery_rate, 2)

    def test_precision(self):
        self.assertAlmostEqual(self.df_binary.da.precision.loc[0], 0.83, 2)
        self.assertAlmostEqual(self.df_binary.da.micro_precision, 0.80, 2)
        self.assertAlmostEqual(self.df_multi.da.precision.loc['setosa'], 1.0, 2)
        self.assertAlmostEqual(self.df_multi.da.micro_precision, 0.84, 2)
        self.assertAlmostEqual(self.df_multi.da.micro_precision,
                               1 - self.df_multi.da.micro_false_discovery_rate, 2)

    def test_recall(self):
        self.assertAlmostEqual(self.df_binary.da.recall.loc[0], 0.83, 2)
        self.assertAlmostEqual(self.df_binary.da.micro_recall, 0.80, 2)
        self.assertAlmostEqual(self.df_multi.da.recall.loc['setosa'], 1.0, 2)
        self.assertAlmostEqual(self.df_multi.da.micro_recall, 0.84, 2)
        self.assertAlmostEqual(self.df_multi.da.micro_recall,
                               1 - self.df_multi.da.micro_false_discovery_rate, 2)

    def test_specificity(self):
        self.assertAlmostEqual(self.df_binary.da.specificity.loc[0], 0.75, 2)
        self.assertAlmostEqual(self.df_binary.da.micro_specificity, 0.80, 2)
        self.assertAlmostEqual(self.df_multi.da.specificity.loc['setosa'], 1.0, 2)
        self.assertAlmostEqual(self.df_multi.da.micro_specificity, 0.92, 2)
        self.assertAlmostEqual(self.df_multi.da.micro_specificity,
                               1 - self.df_multi.da.micro_false_positive_rate, 2)

    def test_specificity(self):
        self.assertAlmostEqual(self.df_binary.da.specificity.loc[0], 0.75, 2)
        self.assertAlmostEqual(self.df_binary.da.micro_specificity, 0.80, 2)
        self.assertAlmostEqual(self.df_multi.da.specificity.loc['setosa'], 1.0, 2)
        self.assertAlmostEqual(self.df_multi.da.micro_specificity, 0.92, 2)
        self.assertAlmostEqual(self.df_multi.da.micro_specificity,
                               1 - self.df_multi.da.micro_false_positive_rate, 2)

    def test_true_negative_rate(self):
        self.assertAlmostEqual(self.df_binary.da.true_negative_rate.loc[0], 0.75, 2)
        self.assertAlmostEqual(self.df_binary.da.micro_true_negative_rate, 0.80, 2)
        self.assertAlmostEqual(self.df_multi.da.true_negative_rate.loc['setosa'], 1.0, 2)
        self.assertAlmostEqual(self.df_multi.da.micro_true_negative_rate, 0.92, 2)
        self.assertAlmostEqual(self.df_multi.da.micro_true_negative_rate,
                               1 - self.df_multi.da.micro_false_positive_rate, 2)

    def test_true_positive_rate(self):
        self.assertAlmostEqual(self.df_binary.da.true_positive_rate.loc[0], 0.83, 2)
        self.assertAlmostEqual(self.df_binary.da.micro_true_positive_rate, 0.80, 2)
        self.assertAlmostEqual(self.df_multi.da.true_positive_rate.loc['setosa'], 1.0, 2)
        self.assertAlmostEqual(self.df_multi.da.micro_true_positive_rate, 0.84, 2)
        self.assertAlmostEqual(self.df_multi.da.micro_true_positive_rate,
                               1 - self.df_multi.da.micro_false_negative_rate, 2)
