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
from collections import defaultdict
from sklearn.model_selection._split import KFold
# from sklearn.metrics.classification import (brier_score_loss,
#                                             log_loss as sk_log_loss)
import mock
from sklearn.svm.classes import OneClassSVM
from mock import patch
from click.testing import CliRunner
from sklearn.ensemble.iforest import IsolationForest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.scorer import brier_score_loss_scorer

from sod.core.evaluation import (split, classifier, predict, _predict,
                                 train_test_split, TrainingParam, TestParam)
from sod.core.dataset import open_dataset
from sod.evaluate import run
from sod.core import paths, pdconcat, OUTLIER_COL, PREDICT_COL
from sod.core.metrics import log_loss, confusion_matrix, roc_auc_score,\
    average_precision_score


class Tester:

    datadir = join(dirname(__file__), 'data')

    N = 200
    dfr = pd.DataFrame({
        'psd@5sec': np.random.random(N),
        'psd@2sec': np.random.random(N),
        'outlier': [False] * N
    })
    clf = classifier(IsolationForest, dfr[['psd@2sec', 'psd@5sec']])

#     def test_to_matrix(self):
#         val0 = self.dfr.loc[0, 'magnitude']
#         val1 = self.dfr.loc[0, 'distance_km']
#         data = self.dfr[['magnitude', 'distance_km']].values
#         assert data[0].tolist() == [val0, val1]
#         data = self.dfr[['distance_km', 'magnitude']].values
#         assert data[0].tolist() == [val1, val0]

    def test_traintest(self):
        test_indices = []
        for train, test in train_test_split(self.dfr):
            assert train[train.index.isin(test.index)].empty
            test_indices.extend(test.index.values.tolist())
        expected_indices = sorted(self.dfr.index)
        assert sorted(test_indices) == expected_indices

        # internal test that dataframe.values returns a COPY of the dataframe
        # data, so that we can pass to the fit and predict method dataframe
        # copies, because their data will not be affected
        # are not modifying the input dataframe:
        for train, test in train_test_split(self.dfr):
            break
        vals = train.values
        value = train.iloc[0, 0]
        vals[0][0] = -value
        assert train.iloc[0, 0] == value
        assert train.values[0][0] == -vals[0][0]
        assert vals.flags['OWNDATA'] is False
        assert vals.flags['WRITEBACKIFCOPY'] is False
        
        # now filter
        train_flt = train[train.iloc[:, 0] == value]
        vals2 = train_flt.values
        value2 = train_flt.iloc[0, 0]
        assert value2 == value
        vals2[0][0] = -value2
        assert train_flt.iloc[0, 0] == value
        assert train_flt.values[0][0] == -vals[0][0]
        assert train.iloc[0, 0] == value
        assert train.values[0][0] == -vals[0][0]
        assert vals2.flags['OWNDATA'] is False
        assert vals2.flags['WRITEBACKIFCOPY'] is False

    @pytest.mark.parametrize('size, n_folds', [
        (10, 11),
        (10, 9),
        (10, 10),
        (10, 1),
        (10, 2),
        (14534, 7),
        (14534, 10),
        # (391453, 10),
    ])
    def test_split(self, size, n_folds):
        if n_folds > size or n_folds < 2:
            with pytest.raises(ValueError):
                list(split(size, n_folds))
            return
        a = list(split(size, n_folds))
        assert a[0][0] == 0
        assert a[-1][1] == size
        assert len(a) == n_folds
        for i, elm in enumerate(a[1:], 1):
            assert elm[1] >= elm[0]
            assert elm[0] == a[i-1][1]
            assert np.abs(np.abs(elm[1] - elm[0]) -
                          np.abs(a[i-1][1] - a[i-1][0])) <= 1

        indices = np.arange(size).tolist()
        dfr = pd.DataFrame(index=np.arange(size))
        all_trn = []
        all_tst = []
        for trn, tst in train_test_split(dfr, n_folds):
            assert sorted(set(trn.index) | set(tst.index)) == indices
            all_trn.extend(trn.index)
            all_tst.extend(tst.index)
        assert sorted(set(all_trn)) == indices
        assert sorted(set(all_tst)) == indices

    @patch('sod.core.evaluation._predict')
    def test_logloss(self, mock_predict):
        eps = 1e-15
        MINVAL = -np.log(1 - eps)
        MAXVAL = -np.log(eps)
        mock_predict.side_effect = lambda *a, **kw: np.array([0.0, 1.0])

        dfr = pd.DataFrame([
            {
                'outlier': False,
                'psd@2sec': .0,
                'psd@5sec': 1.
             },
            {
                'outlier': True,
                'psd@2sec': .0,
                'psd@5sec': 1.
             },
        ])
        pred_df = predict(None, dfr, ['psd@2sec', 'psd@5sec'])
        ll = log_loss(pred_df)
        assert np.isclose(ll, MINVAL)

        # test that log_loss converts booleans to floats. Provide floats
        # and see that result is the same:
        dfr = pd.DataFrame([
            {
                'outlier': 0.0,
                'psd@2sec': .0,
                'psd@5sec': 1.
             },
            {
                'outlier': 1.0,
                'psd@2sec': .0,
                'psd@5sec': 1.
             },
        ])
        pred_df = predict(None, dfr, ['psd@2sec', 'psd@5sec'])
        ll = log_loss(pred_df)
        assert np.isclose(ll, MINVAL)
        
        # test that now we have a better log loss.
        # Swap the ground truth labels (True False):
        dfr = pd.DataFrame([
            {
                'outlier': True,
                'psd@2sec': .0,
                'psd@5sec': 1.
             },
            {
                'outlier': False,
                'psd@2sec': .0,
                'psd@5sec': 1.
             },
        ])
        pred_df = predict(None, dfr, ['psd@2sec', 'psd@5sec'])
        ll = log_loss(pred_df)
        assert np.isclose(ll, MAXVAL, rtol=1e-3)

        # test mid-case, should be greater than 0 but lower than max        
        dfr = pd.DataFrame([
            {
                'outlier': False,
                'psd@2sec': .0,
                'psd@5sec': 1.
             },
            {
                'outlier': False,
                'psd@2sec': .0,
                'psd@5sec': 1.
             },
        ])
        pred_df = predict(None, dfr, ['psd@2sec', 'psd@5sec'])
        ll = log_loss(pred_df)
        assert MINVAL < ll < MAXVAL
        
        # same as above, but labels inverted: same log loss as above        
        dfr = pd.DataFrame([
            {
                'outlier': True,
                'psd@2sec': .0,
                'psd@5sec': 1.
             },
            {
                'outlier': True,
                'psd@2sec': .0,
                'psd@5sec': 1.
             },
        ])
        pred_df = predict(None, dfr, ['psd@2sec', 'psd@5sec'])
        ll = log_loss(pred_df)
        assert MINVAL < ll < MAXVAL
        
        # test that log loss is greater for BIG distances
        dfr = pd.DataFrame([
            {
                'outlier': True,
                'psd@2sec': .0,
                'psd@5sec': 1.
             },
            {
                'outlier': True,
                'psd@2sec': .0,
                'psd@5sec': 1.
             },
        ])
        mock_predict.side_effect = lambda *a, **kw: np.array([0.0, 1.0])
        pred_df = predict(None, dfr, ['psd@2sec', 'psd@5sec'])
        ll1 = log_loss(pred_df)
        mock_predict.side_effect = lambda *a, **kw: np.array([0.5, 0.5])
        pred_df = predict(None, dfr, ['psd@2sec', 'psd@5sec'])
        ll2 = log_loss(pred_df)
        assert ll2 < ll1

    @patch('sod.core.evaluation._predict')
    def test_get_scores(self, mock_predict):
        dfr = pd.DataFrame([
            {'outlier': False,'psd@2sec': .0,},
            {'outlier': True, 'psd@2sec': .0,}
        ])

        mock_predict.side_effect = lambda *a, **kw: np.array([0, 1])
        pred_df = predict(None, dfr, ['psd@2sec'])
        cmt = confusion_matrix(pred_df)
        assert cmt.recall.sum() == 2
        assert cmt.f1score.sum() == 2
        assert cmt.precision.sum() == 2

        mock_predict.side_effect = lambda *a, **kw: np.array([1, 0])
        pred_df = predict(None, dfr, ['psd@2sec'])
        cmt = confusion_matrix(pred_df)
        assert cmt.recall.sum() == 0
        assert cmt.f1score.sum() == 0
        assert cmt.precision.sum() == 0

        mock_predict.side_effect = lambda *a, **kw: np.array([0, 0])
        pred_df = predict(None, dfr, ['psd@2sec'])
        cmt = confusion_matrix(pred_df)
        assert np.allclose(cmt.recall, [1, 0])
        assert np.allclose(cmt.f1score, [.666666, 0])
        assert np.allclose(cmt.precision, [.5, 0])

        mock_predict.side_effect = lambda *a, **kw: np.array([1, 1])
        pred_df = predict(None, dfr, ['psd@2sec'])
        cmt = confusion_matrix(pred_df)
        assert np.allclose(cmt.recall, [0, 1])
        assert np.allclose(cmt.f1score, [0, .666666])
        assert np.allclose(cmt.precision, [0, .5])

    def test_get_scores_order(self):
        '''test that scikit predcit preserves oreder, i.e.:
        predict(x1, x2 ...]) == [predict(x1), predict(x2), ...]
        '''
        res = _predict(self.clf,
                       self.dfr.iloc[10:15, :][['psd@2sec', 'psd@5sec']])
        res2 = []
        for _ in range(10, 15):
            _ = _predict(self.clf,
                         self.dfr.iloc[_:_+1, :][['psd@2sec', 'psd@5sec']])
            res2.append(_[0])
        assert (res == res2).all()

    def test_dirs_exist(self):
        '''these tests MIGHT FAIL IF DIRECTORIES ARE NOT YET INITIALIZED
        (no evaluation or stream2segment run)
        JUST CREATE THEM IN CASE
        '''
        filepaths = []
        for _ in dir(paths):
            if not _.startswith('_'):
                val = getattr(paths, _)
                if isinstance(val, (str, bytes)):
                    filepaths.append(val)
        assert len(filepaths) == 3
        assert all(isdir(_) for _ in filepaths)


    def test_roc_aps(self):
        # 1/th of outliers wrongly classified as inliers
        pdf = pd.DataFrame({
            OUTLIER_COL: np.append(np.zeros(10000), np.ones(10000)),
            PREDICT_COL: np.append(np.zeros(11000), np.ones(9000))
        })
        auc = roc_auc_score(pdf)
        aps = average_precision_score(pdf)
        # auc and avs are the same
        assert np.isclose(auc, 0.95)
        assert np.isclose(aps, 0.95)

        # 1/th of inliers wrongly classified as outliers
        pdf = pd.DataFrame({
            OUTLIER_COL: np.append(np.zeros(10000), np.ones(10000)),
            PREDICT_COL: np.append(np.zeros(9000), np.ones(11000))
        })
        auc = roc_auc_score(pdf)
        aps = average_precision_score(pdf)
        # auc is the same avs GETS WORSE
        assert np.isclose(auc, 0.95)
        assert np.isclose(aps, 0.90909)

    @pytest.mark.parametrize('filename,expected_dic', [
        (
            './whatever/clf=IsolationForest&tr_set=a%2F&feats=f1,f2%2C&b=2.5&c=3@{%26',
            {
                'clf': ('IsolationForest',),
                'b': ('2.5',),
                'c': ('3@{&',),
                'tr_set': ('a/',),
                'feats': ('f1', 'f2,')
            }
        ),
        (
            './whatever/clf=IsolationForest&tr_set=a%2F&feats=f1,f2%2C&b=2.5&c=3@{%26.sklmodel',
            {
                'clf': ('IsolationForest',),
                'b': ('2.5',),
                'c': ('3@{&',),
                'tr_set': ('a/',),
                'feats': ('f1', 'f2,')
            }
        ),
        (
            'clf=IsolationForest&tr_set=a%2F&feats=f1,f2%2C&b=2.5&c=3@{%26',
            {
                'clf': ('IsolationForest',),
                'b': ('2.5',),
                'c': ('3@{&',),
                'tr_set': ('a/',),
                'feats': ('f1', 'f2,')
            }
        ),
    ])
    def test_paramencdec(self, filename, expected_dic):
        dic = TestParam.model_params(filename)
        vals = [dic[k] for k in sorted(dic)]
        expected_vals = [expected_dic[k] for k in sorted(expected_dic)]
        for v1, v2 in zip(vals, expected_vals):
            if v1 != v2:
                asd = 9
        assert all(v1 == v2 for v1, v2 in zip(vals, expected_vals))

        expected_file = basename(filename)
        if not expected_file.endswith('.sklmodel'):
            expected_file += '.sklmodel'
        assert TrainingParam.model_filename(IsolationForest, dic['tr_set'][0],
                                            *dic['feats'], b=dic['b'],
                                            c=dic['c']) == expected_file
        # different params order (keyword arguments), same result:
        assert TrainingParam.model_filename(IsolationForest, dic['tr_set'][0],
                                            *dic['feats'], c=dic['c'],
                                            b=dic['b']) == expected_file
        
#         strs = ['bla/asd?f=1,2&c=ert',
#                 'asd?f=1,2&c=ert',
#                 '?f=1,2&c=ert',
#                 'f=1,2&c=ert']
#         for _ in list(strs):
#             strs.append(_+'.extension')
#             strs.append(_.replace('f=', 'features='))
#         for str_ in strs:
#             if '?' not in str_:
#                 with pytest.raises(ValueError):
#                     TestParam.model_params(str_)
#                 continue
#             dic = TestParam.model_params(str_)
#             if 'features=' in str_:
#                 assert sorted(dic.keys()) == ['c', 'features']
#                 assert dic['features'] == ('1', '2')
#             else:
#                 assert sorted(dic.keys()) == ['c', 'f']
#                 assert dic['f'] == '1,2'
#             assert dic['c'] == \
#                 'ert' + ('.extension' if '.extension' in str_ else '')
# 
#         # try with a string containing a comma percent encoded (%2C):
#         str_ = 'bla/asd?f=1%2C2&c=ert'
#         dic = TestParam.model_params(str_)
#         assert sorted(dic.keys()) == ['c', 'f']
#         assert dic['f'] == '1,2'
#         assert dic['c'] == 'ert'
#         assert TrainingParam.model_filename(**dic) == str_[str_.index('?'):]
#         assert TrainingParam.model_filename(3, 's,', **dic) == \
#             '?features=3,s%2C&f=1%2C2&c=ert'
