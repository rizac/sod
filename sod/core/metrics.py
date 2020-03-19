'''
Created on 27 Feb 2020

@author: riccardo
'''
import types
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix as scikit_confusion_matrix,
    log_loss as scikit_log_loss,
    roc_curve as scikit_roc_curve,
    precision_recall_curve as scikit_pr_curve,
    auc as scikit_auc,
    roc_auc_score as scikit_roc_auc_score,
    average_precision_score as scikit_average_precision_score,
    precision_recall_fscore_support
)
from sod.core import OUTLIER_COL, PREDICT_COL, CLASSNAMES


def log_loss(predicted_df, eps=1e-15, return_mean=True):
    '''Computes the log loss of `predicted_df`

    :param predicted_df: A dataframe with predictions, the output of
        `predict`
    :param return_mean: bool, optional (default=True)
        If true, return the mean loss per sample.
        Otherwise, return the sum of the per-sample losses.

    :return: a NUMBER representing the mean (normalize=True) or sum
        (normalize=False) of all scores in predicted_df
    '''
    return scikit_log_loss(predicted_df[OUTLIER_COL],
                           predicted_df[PREDICT_COL],
                           eps=eps, normalize=return_mean,
                           labels=[False, True])


def confusion_matrix(predicted_df, threshold=0.5, compute_eval_metrics=True):
    '''Returns a pandas dataframe representing the confusion matrix of the
    given predicted dataframe.
    The dataframe will also have the methods 'precision', 'recall' 'f1score'
    and 'support' (number of instances) all returning a two element array
    relative to the inlier class (index 0) and outlier class (index 1)

    :param predicted_df: pandas DataFrame. MUST have the columns
        'outlier' (boolean or boolean like, e.g. 0s and 1s) and
        'predicted_anomaly_score' (floats between 0 and 1, bounds included).
        'outlier' represents the true class label (`y_true` in many
        scikit learn function), whereas predicted_anomaly_score represents
        the prediction of a given classifier (`y_pred` or `y_score` in many
        scikit learn functions)
    '''
    y_pred = predicted_df[PREDICT_COL] > threshold
    return _confusion_matrix(predicted_df.outlier,
                             y_pred,
                             labels=[False, True],
                             compute_eval_metrics=compute_eval_metrics)


def _confusion_matrix(y_true, y_pred, labels=None, compute_eval_metrics=False):
    cmx = scikit_confusion_matrix(y_true,
                                  y_pred,
                                  labels=labels)
    dfr = pd.DataFrame(data=cmx, index=CLASSNAMES, columns=CLASSNAMES)
    if compute_eval_metrics:
        prfs = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                               average=None)
        assert (prfs[-1] == dfr.sum(axis=1)).all()
        P, R, F, S = 'precision', 'recall', 'f1score', 'support'
        dfr[S] = prfs[-1]
        dfr[R] = prfs[1]
        dfr[P] = prfs[0]
        dfr[F] = prfs[2]

#         # attach bound methods: https://stackoverflow.com/a/2982
#         # dfr.precision(), dfr.recall() etcetera
#         dfr.p = dfr.precisions = dfr.precision = \
#             types.MethodType(lambda s: s.loc[:, P], dfr)
#         dfr.r = dfr.recalls = dfr.recall = \
#             types.MethodType(lambda s: s.loc[:, R], dfr)
#         dfr.f = dfr.f1scores = dfr.f1score = \
#             types.MethodType(lambda s: s.loc[:, F], dfr)
#         dfr.s = dfr.supports = dfr.support = \
#             types.MethodType(lambda s: s.loc[:, S], dfr)
#         dfr.num_instances = types.MethodType(lambda s: s.loc[:, S], dfr)

    return dfr


def roc_curve(predicted_df):
    '''Computes the ROC curvecurve returning the tuple
    ```
        fpr, tpr, thresholds, index, best_threshold
    ```
     where the first three elements are numeric arrays, and
    `index, best_threshold` are the index and the value of `thresholds`
    maximizing the mean true recognition rate:
    ```
    (tpr + tnr) / 2 = (tpr + (1 - fpr)) / 2
    ```
    **Note that when plotting a ROC curve, TPR is on the y axis**
    '''
    return _binary_clf_curve(predicted_df[OUTLIER_COL],
                             predicted_df[PREDICT_COL], method='roc')


def precision_recall_curve(predicted_df):
    '''Computes the Precision/Recall curve returning the tuple
    ```
        prec, rec, thresholds, index, best_threshold
    ```
    where the first three elements are numeric arrays, and
    `index, best_threshold` are the index and the value of `thresholds`
    maximizing the F1Score:
    ```
        2 * prec * rec / (prec + rec)
    ```
    **Note that when plotting a PR curve, Precision is on the y axis**
    '''
    return _binary_clf_curve(predicted_df[OUTLIER_COL],
                             predicted_df[PREDICT_COL], method='pr')


def _binary_clf_curve(y_true, y_score, method='roc'):
    if method == 'roc':
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve

        fpr, tpr, thresholds = scikit_roc_curve(y_true, y_score, pos_label=1)
        # Convert to TNR (avoid dividing by 2 as useless):
        tnr = 1 - fpr

        # get the best threshold where we have the best mean of TPR and TNR:
        scores = harmonic_mean(tnr, tpr)

        # Get tbest threshold ignoring 1st score. From the docs (see linke
        # above): thresholds[0] represents no instances being predicted and
        # is arbitrarily set to max(y_score) + 1.
        best_th_index = 1 + np.argmax(scores[1:])
        return fpr, tpr, thresholds, best_th_index, thresholds[best_th_index]

    if method == 'pr':
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve

        prc, rec, thresholds = scikit_pr_curve(y_true, y_score, pos_label=1)

        # get the best threshold where we have the best F1 score
        # (avoid multiplying by 2 as useless):
        # also , consider avoiding numpy warning for NaNs:
        scores = harmonic_mean(prc, rec)

        # Get best score ignoring lat score. From the docs (see link above):
        # the last precision and recall values are 1. and 0. respectively and
        # do not have a corresponding threshold. This ensures that the graph
        # starts on the y axis.
        best_th_index = np.argmax(scores[:-1])
        return prc, rec, thresholds, best_th_index, thresholds[best_th_index]

    raise ValueError('`method` argument in `best_threshold` must be '
                     'either "roc" (ROC curve) or '
                     '"pr" (Precision-Recall Curve)')


def harmonic_mean(x, y):
    '''Computes (element-wise) the harmonic mean of x and y'''
    if len(x) != len(y):
        raise ValueError('Harmonic mean can be calculated on equally sized '
                         'arrays, arrays lengths are %d and %d' %
                         (len(x), len(y)))
    scores = np.zeros(len(x), dtype=float)
    isfinite = (x != 0) | (y != 0)
    xfinite, yfinite = x[isfinite], y[isfinite]
    scores[isfinite] = 2 * (xfinite * yfinite) / (xfinite + yfinite)
    return scores


def roc_auc_score(predicted_df):
    '''Computes the area under the ROC curve

    :param y_true: the labels, in {0, 1} or {True, False} (True or 1 = outlier)
    :param y_score: the scores, or predictions from a given classifier. All
        number should be real numbers in [0, 1] (>=0 and <=1)
    '''
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc
    return scikit_roc_auc_score(predicted_df[OUTLIER_COL],
                                predicted_df[PREDICT_COL])
                                # pos_label=1)


def average_precision_score(predicted_df):
    '''Computes the average prediction score, which is the `auc` equivalent
    of the Precision-Recall curve

    :param y_true: the labels, in {0, 1} or {True, False} (True or 1 = outlier)
    :param y_score: the scores, or predictions from a given classifier. All
        number should be real numbers in [0, 1] (>=0 and <=1)
    '''
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
    return scikit_average_precision_score(predicted_df[OUTLIER_COL],
                                          predicted_df[PREDICT_COL],
                                          pos_label=1)
