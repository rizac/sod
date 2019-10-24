'''
Created on 11 Oct 2019

@author: riccardo
'''
import numpy as np
import pytest
from os import makedirs
from os.path import join, abspath, dirname, isdir, isfile
import pandas as pd
import shutil
from itertools import repeat
from sod.evaluation import pdconcat  #, train_test_split
from collections import defaultdict
from sklearn.model_selection._split import KFold
from sod.evaluation import split, cmatrix, classifier, predict, _predict, open_dataset,\
    Evaluator, train_test_split
from sklearn.metrics.classification import confusion_matrix
import mock
from sklearn.svm.classes import OneClassSVM
from sod.evaluation.ocsvm import OcsvmEvaluator


class Tester:

    dfr = open_dataset(join(dirname(__file__), '..', 'sod', 'dataset',
                       'dataset.secondtry.hdf'), False)

    clf = classifier(OneClassSVM, dfr.iloc[:5,:][['delta_pga', 'delta_pgv']])

    tmpdir = join(dirname(__file__), 'tmp')

    def test_to_matrix(self):
        val0 = self.dfr.loc[0, 'magnitude']
        val1 = self.dfr.loc[0, 'distance_km']
        data = self.dfr[['magnitude', 'distance_km']].values
        assert data[0].tolist() == [val0, val1]
        data = self.dfr[['distance_km', 'magnitude']].values
        assert data[0].tolist() == [val1, val0]

    def test_traintest(self):
        test_indices = []
        for train, test in train_test_split(self.dfr):
            assert train[train.index.isin(test.index)].empty
            test_indices.extend(test.index.values.tolist())
        expected_indices = sorted(self.dfr.index)
        assert sorted(test_indices) == expected_indices

    @pytest.mark.parametrize('size, n_folds', [
        (10, 11),
        (10, 9),
        (10, 10),
        (10, 1),
        (14534, 7),
        (14534, 10),
        (3914534, 10),
    ])
    def test_split(self, size, n_folds):
        a = list(split(size, n_folds))
        assert a[0][0] == 0
        assert a[-1][1] == size
        assert len(a) == n_folds
        for i, elm in enumerate(a[1:], 1):
            assert elm[1] >= elm[0]
            assert elm[0] == a[i-1][1]
            assert np.abs(np.abs(elm[1] - elm[0]) -
                          np.abs(a[i-1][1] - a[i-1][0])) <= 1

    @mock.patch('sod.evaluation._predict')
    def test_get_scores(self, mock_predict):
        dfr = pd.DataFrame([{'outlier': 0, 'modified': '', 'Segment.db.id': 1},
                            {'outlier': 1, 'modified': 'invchanged', 'Segment.db.id': 2}])
        mock_predict.side_effect = lambda *a, **kw: np.array([1, -1])
        pred_df = predict(None, dfr)
        assert pred_df['correctly_predicted'].sum() == 2
        cm_ = cmatrix(pred_df)
        cm_ok_row = cm_.loc['ok', :]
        cm_outlier_row = cm_.loc['outlier', :]
        assert (cm_ok_row == [1, 0]).all()
        assert (cm_outlier_row == [0, 1]).all()

        dfr = pd.DataFrame([{'outlier': 0, 'modified': '', 'Segment.db.id': 1},
                            {'outlier': 0, 'modified': '', 'Segment.db.id': 1},
                            {'outlier': 1, 'modified': 'invchanged', 'Segment.db.id': 1}])
        mock_predict.side_effect = lambda *a, **kw: np.array([1, -1, -1])
        pred_df = predict(None, dfr)
        assert pred_df['correctly_predicted'].sum() == 2
        cm_ = cmatrix(pred_df)
        cm_ok_row = cm_.loc['ok',:]
        cm_outlier_row = cm_.loc['outlier',:]
        assert (cm_ok_row == [1, 1]).all()
        assert (cm_outlier_row == [0, 1]).all()

        dfr = pd.DataFrame([{'outlier': 0, 'modified': '', 'Segment.db.id': 1},
                            {'outlier': 0, 'modified': '', 'Segment.db.id': 1},
                            {'outlier': 1, 'modified': 'invchanged', 'Segment.db.id': 3}])
        mock_predict.side_effect = lambda *a, **kw: np.array([1, -1, -1])
        pred_df = predict(None, dfr)
        assert pred_df['correctly_predicted'].sum() == 2
        cm_ = cmatrix(pred_df)
        cm_ok_row = cm_.loc['ok',:]
        cm_outlier_row = cm_.loc['outlier',:]
        assert (cm_ok_row == [1, 1]).all()
        assert (cm_outlier_row == [0, 1]).all()

    def test_get_scores_order(self):
        '''test that scikit predcit preserves oreder, i.e.:
        predict(x1, x2 ...]) == [predict(x1), predict(x2), ...]
        '''
        res = _predict(self.clf,
                         self.dfr.iloc[10:12, :][['delta_pga', 'delta_pgv']])
        for _ in [10, 12]:
            res2 = _predict(self.clf,
                           self.dfr.iloc[_:_+1, :][['delta_pga', 'delta_pgv']])
            asd = 9

    def test_evaluator(self,
                       # pytest fixutres:
                       #tmpdir
                       ):
        if isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)
        makedirs(self.tmpdir)
        root = self.tmpdir
        
        eval = OcsvmEvaluator(
            parameters={'kernel': ['rbf'], 'gamma': ['auto', 10.00]},
            rootoutdir=root,
            n_folds=5
        )
        
#         eval = Evaluator(OneClassSVM,
#                          parameters={'kernel': ['rbf'], 'gamma': ['auto', 1.11]},
#                          n_folds=5,
#                          rootoutdir=root)

        with pytest.raises(ValueError) as verr:
            # not enough test instances with current cv
            eval.run(
                self.dfr.iloc[:20, :],
                columns=[['delta_pgv'], ['delta_pga', 'delta_pgv']]
            )
        eval.run(
            self.dfr.iloc[:50, :],
            columns=[['delta_pgv'], ['delta_pga', 'delta_pgv']]
        )
        
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
        