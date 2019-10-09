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
            raise AttributeError(
                "The input DataFrame must be an n x n square DataFrame"
            )
        if not all([dt == int for dt in obj.dtypes]):
            raise AttributeError(
                "The input DataFrame must contain only integers, but non-ints were detected."
            )
        if not (obj >= 0).all(axis=None):
            raise AttributeError(
                "The input DataFrame must contain all positive integers, but negative "
                "ints were detected."
            )

    @staticmethod
    def _calculate_metrics(obj):
        """
        Count the positive and negative instances for the actual and predicted class.
        Code very slightly modified from this StackOverflow answer by user lucidv01d,
        https://stackoverflow.com/users/576134/lucidv01d
        https://stackoverflow.com/a/43331484/4541548

        :param obj: A pandas DataFrame
        :return: Counts for true positive, false positive, false negative, and true negative
        """
        FP = obj.sum(axis=0) - np.diag(obj)
        FN = obj.sum(axis=1) - np.diag(obj)
        TP = pd.Series(np.diag(obj), index=obj.index)
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
        return 2 * (
            (self.precision * self.sensitivity) / (self.precision + self.sensitivity)
        )

    @property
    def false_discovery_rate(self):
        """False discovery rate is defined as false positive / (true positive + false positive)"""
        return self.FP / (self.TP + self.FP)

    @property
    def false_negative_rate(self):
        """False negative rate  is defined as false negative / (false negative + true positive)"""
        return self.FN / (self.FN + self.TP)

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
    def micro_accuracy(self):
        """Accuracy is defined as (true positive + true negative) / (true positive + false positive + false negative +
         true negative)"""
        return (self.TP.sum() + self.TN.sum()) / (
            self.TP.sum() + self.FP.sum() + self.FN.sum() + self.TN.sum()
        )

    @property
    def micro_f1(self):
        """F1 is the harmonic mean of precision and sensitivity"""
        return 2 * (
            (self.micro_precision * self.micro_sensitivity)
            / (self.micro_precision + self.micro_sensitivity)
        )

    @property
    def micro_false_discovery_rate(self):
        """False discovery rate is defined as false positive / (true positive + false positive)"""
        return self.FP.sum() / (self.TP.sum() + self.FP.sum())

    @property
    def micro_false_negative_rate(self):
        """False negative rate  is defined as false negative / (false negative + true positive)"""
        return self.FN.sum() / (self.FN.sum() + self.TP.sum())

    @property
    def micro_false_positive_rate(self):
        """False positive rate is defined as false positive / (false positive + true negative)"""
        return self.FP.sum() / (self.FP.sum() + self.TN.sum())

    @property
    def micro_negative_predictive_value(self):
        """Negative predictive value is defined as true negative / (true negative + false negative)"""
        return self.TN.sum() / (self.TN.sum() + self.FN.sum())

    @property
    def micro_positive_predictive_value(self):
        """Positive predictive value is defined as true positive / (true positive + false negative)"""
        return self.TP.sum() / (self.TP.sum() + self.FN.sum())

    @property
    def micro_precision(self):
        """Precision is defined as true positive / (true positive + false negative)"""
        return self.TP.sum() / (self.TP.sum() + self.FP.sum())

    @property
    def micro_recall(self):
        """Recall is defined as true Positive / (true positive + false negative)"""
        return self.TP.sum() / (self.TP.sum() + self.FN.sum())

    @property
    def micro_sensitivity(self):
        """Sensitivity is defined as true Positive / (true positive + false negative)"""
        return self.TP.sum() / (self.TP.sum() + self.FN.sum())

    @property
    def micro_specificity(self):
        """Specificity is defined as true negative / (true negative + false positive)"""
        return self.TN.sum() / (self.TN.sum() + self.FP.sum())

    @property
    def micro_true_negative_rate(self):
        """true negative Rate is defined as true negative / (true negative + false positive)"""
        return self.TN.sum() / (self.TN.sum() + self.FN.sum())

    @property
    def micro_true_positive_rate(self):
        """True positive rate is defined as true positive / true positive + false negative"""
        return self.TP.sum() / (self.TP.sum() + self.FN.sum())

    def export_metrics(self, metrics_to_include=None):
        """Returns a DataFrame of all metrics defined in metrics.py
        :param metrics_to_include: list of metrics to include in the summary output (must be defined in metrics.py)
        :return: pandas DataFrame
        """
        if metrics_to_include is None:
            metrics_to_include = __all_metrics__
        return pd.DataFrame(
            {metric: getattr(self, metric) for metric in metrics_to_include}
        ).T.join(
            pd.DataFrame.from_dict(
                {
                    metric: getattr(self, "micro_{}".format(metric))
                    for metric in metrics_to_include
                },
                orient="index",
                columns=["micro-average"],
            )
        )
