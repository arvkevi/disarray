import pandas as pd
import numpy as np

from .metrics import __all_metrics__
from .version import __version__


@pd.api.extensions.register_dataframe_accessor("da")
class PandasConfusionMatrix:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self.TP, self.FP, self.FN, self.TN = self._calculate_metrics(pandas_obj)

    @staticmethod
    def _validate(obj):
        # verify the input DataFrame is square
        x, y = obj.shape
        if x != y:
            raise AttributeError("The input DataFrame must be an n x n square DataFrame")
        if not all([dt == int for dt in obj.dtypes]):
            raise AttributeError("The input DataFrame must contain only integers, but non-ints were detected.")
        if not all(obj >= 0):
            raise AttributeError("The input DataFrame must contain all positive integers, but negative "
                                 "ints were detected.")

    @staticmethod
    def _calculate_metrics(obj):
        """
        Count the positive and negative instances for the actual and predicted class.
        Code from: https://stackoverflow.com/a/43331484/4541548
        :param obj: A pandas DataFrame
        :return: Counts for true positive, false positive, false negative, and true negative
        """
        FP = obj.sum(axis=0) - np.diag(obj)
        FN = obj.sum(axis=1) - np.diag(obj)
        TP = np.diag(obj)
        TN = obj.values.sum() - (FP + FN + TP)
        return TP, FP, FN, TN

    @property
    def accuracy(self):
        """Accuracy is defined as (true positive + true negative) / (true positive + false positive + false negative +
         true negative)"""
        return (self.TP + self.TN) / (self.TP + self.FP + self.FN + self.TN)

    @property
    def f1(self):
        """F1 is the harmonic mean of precision and sensitivity"""
        return 2 * ((self.precision * self.sensitivity) / (self.precision + self.sensitivity))

    @property
    def false_discovery_rate(self):
        """False discovery rate is defined as false positive / (true positive + false positive)"""
        return self.FP / (self.TP + self.FP)

    @property
    def false_negative_rate(self):
        """Negative predictive value is defined as true negative / (true negative + false negative)"""
        return self.TN / (self.TN + self.FN)

    @property
    def false_positive_rate(self):
        """False positive rate is defined as false positive / (false positive + true negative)"""
        return self.FP / (self.FP + self.TN)

    @property
    def negative_predictive_value(self):
        """Negative predictive value is defined as true negative / (true negative + false negative)"""
        return self.TN / (self.TN + self.FN)

    @property
    def positive_predictive_value(self):
        """Positive predictive value is defined as true positive / (true positive + false negative)"""
        return self.precision

    @property
    def precision(self):
        """Precision is defined as true Positive / (true Positive + false negative)"""
        return self.TP / (self.TP + self.FP)

    @property
    def recall(self):
        """Recall is defined as true Positive / (true Positive + false negative)"""
        return self.sensitivity

    @property
    def sensitivity(self):
        """Sensitivity is defined as true Positive / (true Positive + false negative)"""
        return self.TP / (self.TP + self.FN)

    @property
    def specificity(self):
        """Specificity is defined as true negative / (true negative + false positive)"""
        return self.TN / (self.TN + self.FP)

    @property
    def true_negative_rate(self):
        """true negative Rate is defined as (true negative / true negative + false positive)"""
        return self.specificity

    @property
    def true_positive_rate(self):
        """True Positive Rate is defined as true positive / true Positive + false negative"""
        return self.sensitivity

    @property
    def all_metrics(self):
        return pd.DataFrame({metric: getattr(self, metric) for metric in __all_metrics__}).T
