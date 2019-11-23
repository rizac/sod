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
from sklearn.metrics.classification import (confusion_matrix, brier_score_loss,
                                            log_loss)
import mock
from sklearn.svm.classes import OneClassSVM
from mock import patch
from click.testing import CliRunner
from sklearn.ensemble.iforest import IsolationForest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.scorer import brier_score_loss_scorer

from sod.core.evaluation import (split, classifier, predict, _predict,
                                 CVEvaluator, train_test_split,
                                 drop_duplicates,
                                 keep_cols, drop_na, cmatrix_df, ParamsEncDec,
                                 join_save_evaluation)
from sod.core.dataset import (open_dataset, groupby_stations)
from sod.evaluate import (OcsvmEvaluator, run)
from sod.core import paths


class PoolMocker:
    
    def apply_async(self, func, args=None, kwargs=None, callback=None,
                    error_callback=None):
        try:
            if args[2] == ['psd@2sec'] and 'window_type' in args[1].columns:
                asd = 9
            result = func(*(args or []), **(kwargs or {}))
            if callback:
                callback(result)
        except Exception as exc:
            if error_callback:
                error_callback(exc)
            raise
    
    def imap_unordered(self, func, iterable):
        for arg in iterable:
            yield func(arg)

    def close(self):
        pass
    
    def join(self):
        pass
    
    def terminate(self):
        pass


class Tester:

    datadir = join(dirname(__file__), 'data')

    with patch('sod.core.dataset.DATASETS_DIR', datadir):
        dfr = open_dataset(join(datadir, 'pgapgv.hdf_'), False)
        dfr2 = open_dataset(join(datadir, 'oneminutewindows.hdf_'), False)
    # dfr3 = open_dataset('magnitudeenergy.hdf', normalize=False)

    clf = classifier(OneClassSVM, dfr.iloc[:5, :][['delta_pga', 'delta_pgv']])

    cv_evalconfig = join(dirname(__file__), 'data', 'cv.pgapgv.ocsvm.yaml')
    cv_evalconfig2 = join(dirname(__file__), 'data', 'cv.oneminutewindows.ocsvm.yaml')
    evalconfig = join(dirname(__file__), 'data', 'eval.oneminutewindows.yaml')
    evalconfig2 = join(dirname(__file__), 'data', 'eval.pgapgv.yaml')

    tmpdir = join(dirname(__file__), 'tmp')

#     def test_groupby_stations(self):
#         df2 = groupby_stations(self.dfr)
#         assert len(df2) <= len(self.dfr)

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
    def test_get_scores(self, mock_predict):
        dfr = pd.DataFrame([
            {'outlier': False, 'modified': ''},
            {'outlier': True, 'modified': 'invchanged'}
        ])
        dfr.insert(0, 'pgapgv.id', [1, 2])

        mock_predict.side_effect = lambda *a, **kw: np.array([1, -1])
        pred_df = predict(None, dfr)
        assert pred_df['correctly_predicted'].sum() == 2
        cm_ = cmatrix_df(pred_df)
        assert cm_.loc['ok', :].to_dict() == {
            'ok': 1.0,
            'outlier': 0.0,
            '% rec.': 100.0,
            'Mean log_loss': 0.0
        }
        # all others are zero
        assert cm_.iloc[1:, :].sum().sum() == 0

        # why all other rows were zero?
        # Because 'invchanged' does not match any class. Let's rewrite it:
        dfr = pd.DataFrame([
            {'outlier': False, 'modified': ''},
            {'outlier': True, 'modified': 'INVFILE:'}
        ])
        dfr.insert(0, 'pgapgv.id', [1, 2])
        mock_predict.side_effect = lambda *a, **kw: np.array([1, -1])
        pred_df = predict(None, dfr)
        assert pred_df['correctly_predicted'].sum() == 2
        cm_ = cmatrix_df(pred_df)
        assert cm_.loc['ok', :].to_dict() == {
            'ok': 1.0,
            'outlier': 0.0,
            '% rec.': 100.0,
            'Mean log_loss': 0.0
        }
        assert cm_.iloc[1, :].to_dict() == {
            'ok': 0.0,
            'outlier': 1.0,
            '% rec.': 100.0,
            'Mean log_loss': 0.0
        }
        # all others are zero
        assert cm_.iloc[2:, :].sum().sum() == 0
        
        
        # test the mean_log_loss
        dfr = pd.DataFrame([
            {'outlier': False, 'modified': ''},
            {'outlier': False, 'modified': ''},
            {'outlier': True, 'modified': 'INVFILE:'},
            {'outlier': True, 'modified': 'INVFILE:'}
        ])
        dfr.insert(0, 'pgapgv.id', [1, 2, 3, 4])

        # predictions are:
        # ok, slightly bad, slightly bad, ok
        mock_predict.side_effect = lambda *a, **kw: np.array([1, -.1, .1, -1])
        pred_df_low_logloss = predict(None, dfr)
        cm_low_logloss = cmatrix_df(pred_df_low_logloss)
        # predictions are:
        # ok, highly bad, ok, ok
        mock_predict.side_effect = lambda *a, **kw: np.array([1, -1, -1, -1])
        pred_df_high_logloss = predict(None, dfr)
        cm_high_logloss = cmatrix_df(pred_df_high_logloss)
        
        # prediction 1 is worse concerning correctly predicted:
        assert pred_df_low_logloss['correctly_predicted'].sum() == 2
        assert pred_df_high_logloss['correctly_predicted'].sum() == 3
        # but has a lower log loss (i.e., better):
        assert cm_low_logloss['Mean log_loss'].sum() < \
            cm_high_logloss['Mean log_loss'].sum()

    def test_get_scores_order(self):
        '''test that scikit predcit preserves oreder, i.e.:
        predict(x1, x2 ...]) == [predict(x1), predict(x2), ...]
        '''
        res = _predict(self.clf,
                       self.dfr.iloc[10:15, :][['delta_pga', 'delta_pgv']])
        res2 = []
        for _ in range(10, 15):
            _ = _predict(self.clf,
                         self.dfr.iloc[_:_+1, :][['delta_pga', 'delta_pgv']])
            res2.append(_[0])
        assert (res == res2).all()

    @patch('sod.core.dataset.dataset_path')
    @patch('sod.core.evaluation.Pool',
           side_effect=lambda *a, **v: PoolMocker())
    def test_evaluator(self,
                       # pytest fixutres:
                       #tmpdir
                       mock_pool,
                       mock_dataset_in_path
                       ):
        if isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

        INPATH, OUTPATH = self.datadir, self.tmpdir

        with patch('sod.evaluate.EVALUATIONS_CONFIGS_DIR', INPATH):
            with patch('sod.evaluate.EVALUATIONS_RESULTS_DIR', OUTPATH):

                mock_dataset_in_path.side_effect = \
                    lambda filename, *a, **v: join(INPATH, filename)

                for evalconfigpath in [self.cv_evalconfig, self.cv_evalconfig2]:
                    evalconfigname = basename(evalconfigpath)
                    runner = CliRunner()
                    result = runner.invoke(run, ["-c", evalconfigname])
                    assert not result.exception

                    assert listdir(join(OUTPATH, evalconfigname))
                    # check prediction file:
                    html_file = None
                    prediction_file = None
                    model_file = None
                    subdirs = (CVEvaluator.EVALREPORTDIRNAME,
                               CVEvaluator.PREDICTIONSDIRNAME,
                               CVEvaluator.MODELDIRNAME)
                    assert sorted(listdir(join(OUTPATH, evalconfigname))) == \
                        sorted(subdirs)
                    for subdir in subdirs:
                        assert listdir(join(OUTPATH, evalconfigname, subdir))

                    prediction_file = \
                        listdir(join(OUTPATH, evalconfigname,
                                     CVEvaluator.PREDICTIONSDIRNAME))[0]
                    prediction_file = join(OUTPATH, evalconfigname,
                                           CVEvaluator.PREDICTIONSDIRNAME,
                                           prediction_file)
                    id = basename(evalconfigpath).split('.')[1] + '.id'
                    if evalconfigpath == self.cv_evalconfig:
                        cols = ['correctly_predicted', 'outlier', 'modified',
                                id, 'decision_function',
                                'log_loss']
                    else:
                        cols = ['window_type', 'log_loss', 'decision_function',
                                'correctly_predicted', 'outlier', 'modified',
                                id]
                    assert sorted(pd.read_hdf(prediction_file).columns) == \
                        sorted(cols)

                runner = CliRunner()
                result = runner.invoke(run, ["-c", basename(self.cv_evalconfig2)])
                # directory exists:
                assert result.exception

                # now run evaluations with the generated files:
                runner = CliRunner()
                result = runner.invoke(run, ["-c", basename(self.evalconfig)])
                assert not result.exception
                runner = CliRunner()
                result = runner.invoke(run, ["-c", basename(self.evalconfig2)])
                assert not result.exception

        join_save_evaluation(join(OUTPATH,
                                  basename(self.cv_evalconfig2),
                                  'evalreports'))

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

    def test_drop_cols(self):
        d = pd.DataFrame({
            'a': [1, 4, 5],
            'b': [5, 7.7, 6]
        })
        with pytest.raises(ValueError):
            # dataframe not bound to a dataset
            d_ = keep_cols(d, ['a'])
        
        d['outlier'] = [True, False, False]
        d['modified'] = ['', '', 'inv changed']
        d['pgapgv.id'] = [1, 1, 2]

        with pytest.raises(ValueError):
            # dataframe not bound to a dataset: id col not at first position
            d_ = keep_cols(d, ['a'])

        d.drop('pgapgv.id', axis=1, inplace=True)
        d.insert(0, 'pgapgv.id', [1, 1, 2])
        
        assert sorted(keep_cols(d, ['modified']).columns.tolist()) == \
            sorted(['pgapgv.id', 'modified', 'outlier'])
        assert sorted(keep_cols(d, ['a', 'b']).columns.tolist()) == \
            sorted(d.columns.tolist())
        assert sorted(keep_cols(d, ['b']).columns.tolist()) == \
            sorted(['pgapgv.id', 'b', 'modified', 'outlier'])

    def test_drop_duplicates(self):
        d = pd.DataFrame({
            'pgapgv.id': [1, 2, 3, 4, 5],
            'a': [1, 4, 5, 5.6, -1],
            'b': [5.56, 5.56, 5.56, 5.56, 5.561],
            'c': [5, 7.7, 6, 6, 6],
            'modified': ['', '', 'INVFILE:', '', ''],
            'outlier': [False, True, True, False, False]
        })

        pd.testing.assert_frame_equal(drop_duplicates(d, ['a']), d)
        assert sorted(drop_duplicates(d, ['b']).index.values) == [0, 1, 2, 4]
        assert sorted(drop_duplicates(d, ['b'], 2).index.values) == [0, 1, 2]

    def test_drop_na(self):
        d = pd.DataFrame({
            'pgapgv.id': [1, 2, 3, 4, 5],
            'a': [1, np.inf, 4, 5.6, -1],
            'b': [5.56, 5.56, np.nan, 5.56, 5.561],
            'c': [5, 7.7, 6, 6, 6],
            'd': [np.inf] * 5,
            'e': [-np.inf] * 5,
            'f': [np.nan] * 5,
            'modified': ['', '', 'INVFILE:', '', ''],
            'outlier': [False, True, True, False, False]
        })

        pd.testing.assert_frame_equal(drop_na(d, ['c']), d)
        assert sorted(drop_na(d, ['a']).index.values) == [0, 2, 3, 4]
        assert sorted(drop_na(d, ['b']).index.values) == [0, 1, 3, 4]
        assert sorted(drop_na(d, ['a', 'b']).index.values) == [0, 3, 4]

        for col in ['d', 'e', 'f']:
            with pytest.raises(Exception):
                drop_na(d, [col])

    def test_paramencdec(self):
        strs = ['bla/asd?f=1,2&c=ert',
                'asd?f=1,2&c=ert',
                '?f=1,2&c=ert',
                'f=1,2&c=ert']
        for _ in list(strs):
            strs.append(_+'.extension')
            strs.append(_.replace('f=', 'features='))
        for str_ in strs:
            if '?' not in str_:
                with pytest.raises(ValueError):
                    ParamsEncDec.todict(str_)
                continue
            dic = ParamsEncDec.todict(str_)
            if 'features=' in str_:
                assert sorted(dic.keys()) == ['c', 'features']
                assert dic['features'] == ('1', '2')
            else:
                assert sorted(dic.keys()) == ['c', 'f']
                assert dic['f'] == ('1,2')
            assert dic['c'] == \
                'ert' + ('.extension' if '.extension' in str_ else '')

        # try with a string containing a comma percent encoded (%2C):
        str_ = 'bla/asd?f=1%2C2&c=ert'
        dic = ParamsEncDec.todict(str_)
        assert sorted(dic.keys()) == ['c', 'f']
        assert dic['f'] == ('1,2')
        assert dic['c'] == 'ert'
        assert ParamsEncDec.tostr(**dic) == str_[str_.index('?'):]
        assert ParamsEncDec.tostr(3, 's', **dic) == \
            '?features=3,s&f=1%2C2&c=ert'
