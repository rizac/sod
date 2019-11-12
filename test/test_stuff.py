'''
Created on 11 Oct 2019

@author: riccardo
'''
import numpy as np
import pytest
from os import makedirs, listdir
from os.path import join, abspath, dirname, isdir, isfile, basename, splitext
import pandas as pd
import shutil
from itertools import repeat, product
from sod.evaluation import pdconcat  #, train_test_split
from collections import defaultdict
from sklearn.model_selection._split import KFold
from sod.evaluation import split, classifier, predict, _predict,\
    Evaluator, train_test_split, drop_duplicates, keep_cols, drop_na
from sklearn.metrics.classification import confusion_matrix, brier_score_loss, log_loss
import mock
from sklearn.svm.classes import OneClassSVM
from sod.evaluation.execute import OcsvmEvaluator, run
from mock import patch
from click.testing import CliRunner
from sod.evaluation.datasets import pgapgv, oneminutewindows
from sod.plot import plot, plot_calibration_curve
from sklearn.ensemble.iforest import IsolationForest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.scorer import brier_score_loss_scorer


class Tester:

    datadir = join(dirname(__file__), 'data')

    dfr = pgapgv(join(datadir, 'pgapgv.hdf'), False)
    dfr2 = oneminutewindows(join(datadir, 'oneminutewindows.hdf'), False)

    clf = classifier(OneClassSVM, dfr.iloc[:5,:][['delta_pga', 'delta_pgv']])

    evalconfig = join(dirname(__file__), 'data', 'pgapgv.ocsvm.yaml')
    evalconfig2 = join(dirname(__file__), 'data', 'oneminutewindows.ocsvm.yaml')

    tmpdir = join(dirname(__file__), 'tmp')

    @patch('sod.plot.plt')        
    def test_plot(self, mock_plt):
        plot(self.dfr, 'noise_psd@5sec', 'noise_psd@2sec', axis_lim=.945,
             clfs={'a': self.clf})
        plot_calibration_curve({'a': self.clf}, self.dfr,
                               ['noise_psd@5sec', 'noise_psd@2sec'])
        # plot_decision_func_2d(None, self.clf)
        
    def tst_calibration(self):
        '''this basically tests CalibratedClassifierCV, and the result is
        that we can not use it with OneClassSVM or IsolationForest. The hack
        here below is to set the attribute classes_ on the evaluator instance,
        but this does not work woth isotonic regression (maybe too few samples?)
        '''
        xtrain = np.random.rand(1000, 2)
        xtest = list(product(range(5, 50), range(5, 50)))  # [[100, 100], [-100, 100], [100, -100], [-100, -100]]

        iforest = IsolationForest(contamination=0)
        iforest.fit(xtrain)
        
        ocsvm = OneClassSVM()
        ocsvm.fit(xtrain)
        
        iforest.classes_ = [False, True]
        ocsvm.classes_ = [False, True]
        
        assert not hasattr(iforest, 'predict_proba')
        assert not hasattr(ocsvm, 'predict_proba')
        
        # Now the problem: CalibratedClassifierCV expects estimators
        # with the classes_ attribute, which ocsvm and iforest don't have.
        # We need to "manually" use platt and isotonic regression algorithms
        # For info see:
        # http://fastml.com/classifier-calibration-with-platts-scaling-and-isotonic-regression/
        
        iforest_dfn = iforest.decision_function(xtest)
        ocsvm_dfn = ocsvm.decision_function(xtest)
        
        iforest_sig= CalibratedClassifierCV(iforest, cv='prefit', method='sigmoid')
        ocsvm_sig = CalibratedClassifierCV(ocsvm, cv='prefit', method='sigmoid')
        
        assert hasattr(iforest_sig, 'predict_proba')
        assert hasattr(ocsvm_sig, 'predict_proba')
        
        iforest_iso= CalibratedClassifierCV(iforest, cv='prefit', method='isotonic')
        ocsvm_iso = CalibratedClassifierCV(ocsvm, cv='prefit', method='isotonic')
        
        assert hasattr(iforest_sig, 'predict_proba')
        assert hasattr(ocsvm_sig, 'predict_proba')
        
        ytrue = [True]*len(xtest) + [False]
        xtest += [[0.5, 0.5]]
        iforest_sig.fit(xtest, ytrue)
        ocsvm_sig.fit(xtest, ytrue)
        iforest_iso.fit(xtest, ytrue)
        ocsvm_iso.fit(xtest, ytrue)

        iforest_sig_dfn = iforest_sig.predict_proba(xtest)
        ocsvm_sig_dfn = ocsvm_sig.predict_proba(xtest)
        iforest_iso_dfn = iforest_iso.predict_proba(xtest)
        ocsvm_iso_dfn = ocsvm_iso.predict_proba(xtest)

        asd = 9
        
    
    def test_scores(self):
        
        
        ytrue = [False]
        ypred=[
            [1, 0]
        ]
        
        with pytest.raises(ValueError):
            # yrue contains only one label:
            ll1 = log_loss(ytrue, ypred)

        # Now, ypred is for outlier detection a list of probabilities (or scores)
        # all in [0, .., 1], defining the:
        # [probability_class_0, probability_class_1, ..., probability_classN]
        # Given that our classes are [False, True] (ok, outlier),
        # a probability (or score) of [0, 1] means: 100% outlier,
        # a probability (or score) of [1, 0] means: 100% ok.
        ypred = [
            [1, 0],  # 100% ok (class False)
            [0, 1]   # 100% outlier (class True)
        ]
        ll1 = log_loss([False, True], ypred)
        ll2 = log_loss([True, False], ypred)
        # assert that ll1 is BETTER (log score is lower):
        assert ll1 < ll2
        
        # now try to see if the log gives more penalty to completely wrong
        # classifications:
        
        # only one correctly classified, but both with high score
        # (the correctly classified class predicts a 1, the mislcassified one predicts the
        # wrong class but with a .55 score)
        ll3 = log_loss([False, True],
                       [
                           [0.45, 0.55],  # 55% outlier (misclassified for few points)
                           [0, 1]  # 100% outlier (correctly classified)
                        ])
        # both correctly classified, but both with a very low score:
        ll4 = log_loss([False, True],
                       [
                           [0.55, 0.45],  # 55% ok (correctly classified, but for few points)
                           [0.45, .55]   # 100% outlier (correctly classified, but for few points)
                        ])
        
        assert ll3 < ll4
        assert ll1 < ll3 < ll4 < ll2
        
        # assert that the 
        y_true = [False, True, True, False]
        y_pred = [
            [0, 1],
            [0.45, 0.55],
            [0.55, 0.45],
            [1, 0]
        ]
        ll1 = log_loss(y_true, y_pred)
        y_pred2 = [
            [1, 0],
            [0.55, 0.45],
            [0.45, 0.55],
            [0, 1]
        ]
        ll2 = log_loss(y_true, y_pred2)
        asd = 9
#     def test_dropduplicates(self):
#         dfr = open_dataset(join(dirname(__file__), '..', 'sod', 'dataset',
#                                 'dataset.hdf'), False)
#         columns = ['magnitude', 'distance_km', 'amp@0.5hz', 'amp@1hz',
#                    'amp@2hz', 'amp@5hz', 'amp@10hz', 'amp@20hz',
#                    'noise_psd@5sec']
#         df1 = drop_duplicates(dfr, columns, 5, verbose=True)

#         columns = ['delta_pga', 'delta_pgv']
#         df2 = drop_duplicates(dfr, columns, 1, verbose=True)
        
#     def test_cmatrix(self):
#         dfr = pd.DataFrame({
#             'label': [1, -1],
#             'predicted': [1, -1]
#         })
#         
#         cm1 = cmatrix(dfr)
#         cm2 = cmatrix(dfr, [1, 100])
#         asd = 9
        
#     def test_make_bins(self):
#         make_bins(self.dfr, 'distance_km')


# def get_confusion_matrix():
#     confusion_matrix(ground_truths, predicted, labels)


# def kfold(dataframe, n_folds, random=True):
#     for start, end in split(len(dataframe), n_folds):
#         if not random:
#             yield dataframe.iloc[start:end]
#         else:
#             _ = dataframe.copy()
#             dfr = _.sample(n=end-start)
#             yield dfr
#             _ = _[~_.index.isin(dfr.index)]
# 
# 
# def train_test_split(dataframe, n_splits=10, column='magnitude', bins=None):
#     '''Yields the tuple (train, test) `n_splits` times. Similar to scikit's
#     `train_test_split` but yields tuples of (disjoint) pandas `DataFrame`s.
#     Moreover, this method does not select dataframe rows randomly, but tries
#     to preserve in each yielded dataframe the distribution of values of the
#     given dataframe (The distribution is built by
#     grouping the `column` values according to the given `bins`)
# 
#     :param dataframe: the source dataframe
#     :param n_splits: the number of splits. Integer, default to 10 (as in a
#         10 fold CV)
#     :param column: string, default: 'magnitude'. The column,
#         denoting the way elements are yielded, by assuring that each yielded
#         dataframe preserves somehow the same distribution under this column
#     :param bins: increasing integer array denoting the lower bound of each
#         group (or bin) whereby the distribution of the dataframe values under
#         the given column is built. If None (the default), it is the sequence
#         of integers from 0 to 9
#     '''
#     if bins is None:
#         bins = np.arange(0, 10, 1)
#     # _mg = pd.cut(dfr['magnitude'], np.array([-5, 2.9, 3.7, 10]), right=False)
#     dfr_series = pd.cut(dataframe[column], bins, right=False)
#     iterators = list(zip(*[_iter(sub_dfr, n_splits)
#                      for _, sub_dfr in dataframe.groupby(dfr_series)]))
#     for dframes in iterators:
#         test = pdconcat(dframes)
#         yield dataframe[~dataframe.index.isin(test.index)], test
# 
# 
# def _iter(dataframe, n_splits=10):
#     '''returns an iterable yielding subset of the given dataframe
#     in `n_folds` chunks. Each row of the given dataframe is present in only one
#     of the yielded chunks'''
#     if dataframe.empty:
#         return repeat(dataframe, n_splits)
#     step = np.true_divide(len(dataframe), n_splits)
#     return (dataframe[np.ceil(step*i).astype(int):
#                       np.ceil(step*(i+1)).astype(int)]
#             for i in range(n_splits))
# 
# 
# def make_bins(dataframe, column, numbins=10):
#     min_, max_ = np.nanmin(dataframe[column]), np.nanmax(dataframe[column])
#     step = (max_ - min_) / (10 * numbins)
#     bins = np.arange(min_, max_ + step, step)
#     expected_step = np.true_divide(len(dataframe), numbins)
#     intervals = []
#     last_interval = None
#     last_interval_count = 0
#     for interval, dfr in dataframe.groupby(pd.cut(dataframe[column], bins,
#                                                   right=False)):
#         size = len(dfr)
#         if last_interval is None:
#             last_interval = interval
#             last_interval_count = 0
#         last_interval_count += len(dfr)
#         if last_interval_count > expected_step:
#             intervals.append(pd.Interval(last_interval.left,
#                                          interval.right, closed='left'))
#             last_interval = None
#     if last_interval is not None:
#         intervals.append(pd.Interval(last_interval.left,
#                                      interval.right, closed='left'))
#     return intervals
        