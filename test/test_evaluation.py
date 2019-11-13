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
                                 Evaluator, train_test_split, drop_duplicates,
                                 keep_cols, drop_na, cmatrix_df)
from sod.core.dataset import (open_dataset, groupby_stations,
                              datasets_input_dir as dataset_datasets_input_dir)
from sod.core.plot import plot, plot_calibration_curve
from sod.evaluate import (OcsvmEvaluator, run,
                          inputcfgpath as evaluate_inputcfgpath,
                          outputpath as evaluate_outputpath)


class Tester:

    datadir = join(dirname(__file__), 'data')

    dfr = open_dataset(join(datadir, 'pgapgv.hdf_'), False)
    dfr2 = open_dataset(join(datadir, 'oneminutewindows.hdf_'), False)

    clf = classifier(OneClassSVM, dfr.iloc[:5, :][['delta_pga', 'delta_pgv']])

    evalconfig = join(dirname(__file__), 'data', 'pgapgv.ocsvm.yaml')
    evalconfig2 = join(dirname(__file__), 'data', 'oneminutewindows.ocsvm.yaml')

    tmpdir = join(dirname(__file__), 'tmp')

    def test_groupby_stations(self):
        df2 = groupby_stations(self.dfr)
        assert len(df2) <= len(self.dfr)

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

    @patch('sod.core.evaluation._predict')
    def test_get_scores(self, mock_predict):
        dfr = pd.DataFrame([
            {'outlier': False, 'modified': '', 'id': 1},
            {'outlier': True, 'modified': 'invchanged', 'id': 2}
        ])
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
            {'outlier': False, 'modified': '', 'id': 1},
            {'outlier': True, 'modified': 'INVFILE:', 'id': 2}
        ])
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
            {'outlier': False, 'modified': '', 'id': 1},
            {'outlier': False, 'modified': '', 'id': 2},
            {'outlier': True, 'modified': 'INVFILE:', 'id': 3},
            {'outlier': True, 'modified': 'INVFILE:', 'id': 4}
        ])
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

#         assert (cm_outlier_row == [0, 1]).all()
# 
#         dfr = pd.DataFrame([{'outlier': False, 'modified': '', 'id': 1},
#                             {'outlier': False, 'modified': '', 'id': 1},
#                             {'outlier': True, 'modified': 'invchanged', 'id': 1}])
#         mock_predict.side_effect = lambda *a, **kw: np.array([1, -1, -1])
#         pred_df = predict(None, dfr)
#         assert pred_df['correctly_predicted'].sum() == 2
#         cm_ = cmatrix_df(pred_df)
#         cm_ok_row = cm_.loc['ok', :]
#         cm_outlier_row = cm_.loc['outlier', :]
#         assert (cm_ok_row == [1, 1]).all()
#         assert (cm_outlier_row == [0, 1]).all()
# 
#         dfr = pd.DataFrame([{'outlier': False, 'modified': '', 'id': 1},
#                             {'outlier': False, 'modified': '', 'id': 1},
#                             {'outlier': True, 'modified': 'invchanged', 'Segment.db.id': 3}])
#         mock_predict.side_effect = lambda *a, **kw: np.array([1, -1, -1])
#         pred_df = predict(None, dfr)
#         assert pred_df['correctly_predicted'].sum() == 2
#         cm_ = cmatrix_df(pred_df)
#         cm_ok_row = cm_.loc['ok', :]
#         cm_outlier_row = cm_.loc['outlier', :]
#         assert (cm_ok_row == [1, 1]).all()
#         assert (cm_outlier_row == [0, 1]).all()

#         # test that we do not need Segment.db.id:
#         dfr = pd.DataFrame([{'outlier': False, 'modified': ''},
#                             {'outlier': False, 'modified': ''},
#                             {'outlier': True, 'modified': 'invchanged'}])
#         mock_predict.side_effect = lambda *a, **kw: np.array([1, -1, -1])
#         pred_df = predict(None, dfr)
#         assert pred_df['correctly_predicted'].sum() == 2
#         cm_ = cmatrix(pred_df)
#         cm_ok_row = cm_.loc['ok', :]
#         cm_outlier_row = cm_.loc['outlier', :]
#         assert (cm_ok_row == [1, 1]).all()
#         assert (cm_outlier_row == [0, 1]).all()

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
    @patch('sod.evaluate.inputcfgpath')
    @patch('sod.evaluate.outputpath')
    def test_evaluator(self,
                       # pytest fixutres:
                       #tmpdir
                       mock_out_path,
                       mock_inputcfgpath,
                       mock_dataset_in_path
                       ):
        if isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)
        INPATH, OUTPATH = self.datadir, self.tmpdir
        mock_out_path.return_value = OUTPATH
        mock_inputcfgpath.return_value = INPATH
        mock_dataset_in_path.side_effect = \
            lambda filename, *a, **v: join(INPATH, filename)

        for evalconfigpath in [self.evalconfig, self.evalconfig2]:
            evalconfigname = basename(evalconfigpath)
            runner = CliRunner()
            result = runner.invoke(run, ["-c", evalconfigname])
            assert not result.exception
    
            assert listdir(join(OUTPATH, evalconfigname))
            # check prediction file:
            html_file=None
            prediction_file = None
            model_file = None
            for fle in listdir(join(OUTPATH, evalconfigname)):
                if not model_file and splitext(fle)[1] == '.model':
                    model_file = join(OUTPATH, evalconfigname, fle)
                elif not prediction_file and splitext(fle)[1] == '.hdf':
                    prediction_file = join(OUTPATH, evalconfigname, fle)
                elif not html_file and splitext(fle)[1] == '.html':
                    html_file = join(OUTPATH, evalconfigname, fle)
            assert html_file and prediction_file and model_file
            
            if evalconfigpath == self.evalconfig:
                cols = ['correctly_predicted', 'outlier', 'modified', 'id',
                        'log_loss']
            else:
                cols = ['window_type', 'log_loss',
                        'correctly_predicted', 'outlier', 'modified', 'id']
            assert sorted(pd.read_hdf(prediction_file).columns) == \
                sorted(cols)
        
        # shutil.rmtree(self.tmpdir)
        
        runner = CliRunner()
        result = runner.invoke(run, ["-c", basename(self.evalconfig2)])

    def test_dirs_exist(self):
        '''these tests MIGHT FAIL IF DIRECTORIES ARE NOT YET INITIALIZED
        (no evaluation or stream2segment run)
        JUST CREATE THEM IN CASE
        '''
        for _ in [dataset_datasets_input_dir,
                  evaluate_inputcfgpath,
                  evaluate_outputpath]:
            assert isdir(_())

    def test_drop_cols(self):
        d = pd.DataFrame({
            'a': [1, 4, 5],
            'b': [5, 7.7, 6]
        })
        d_ = keep_cols(d, ['a'])
        assert len(d_.columns) == 1 and len(d_) == len(d)
        d['outlier'] = [True, False, False]
        d['modified'] = ['', '', 'inv changed']
        d['id'] = [1, 1, 2]

        assert sorted(keep_cols(d, ['modified']).columns.tolist()) == \
            sorted(['id', 'modified', 'outlier'])
        assert sorted(keep_cols(d, ['a', 'b']).columns.tolist()) == \
            sorted(d.columns.tolist())
        assert sorted(keep_cols(d, ['b']).columns.tolist()) == \
            sorted(['id', 'b', 'modified', 'outlier'])

    def test_drop_duplicates(self):
        d = pd.DataFrame({
            'a': [1, 4, 5, 5.6, -1],
            'b': [5.56, 5.56, 5.56, 5.56, 5.561],
            'c': [5, 7.7, 6, 6, 6],
            'modified': ['', '', 'INVFILE:', '', ''],
            'outlier': [False, True, True, False, False],
            'id': [1, 2, 3, 4, 5]
        })

        pd.testing.assert_frame_equal(drop_duplicates(d, ['a']), d)
        assert sorted(drop_duplicates(d, ['b']).index.values) == [0, 1, 2, 4]
        assert sorted(drop_duplicates(d, ['b'], 2).index.values) == [0, 1, 2]

    def test_drop_na(self):
        d = pd.DataFrame({
            'a': [1, np.inf, 4, 5.6, -1],
            'b': [5.56, 5.56, np.nan, 5.56, 5.561],
            'c': [5, 7.7, 6, 6, 6],
            'd': [np.inf] * 5,
            'e': [-np.inf] * 5,
            'f': [np.nan] * 5,
            'modified': ['', '', 'INVFILE:', '', ''],
            'outlier': [False, True, True, False, False],
            'id': [1, 2, 3, 4, 5]
        })

        pd.testing.assert_frame_equal(drop_na(d, ['c']), d)
        assert sorted(drop_na(d, ['a']).index.values) == [0, 2, 3, 4]
        assert sorted(drop_na(d, ['b']).index.values) == [0, 1, 3, 4]
        assert sorted(drop_na(d, ['a', 'b']).index.values) == [0, 3, 4]

        for col in ['d', 'e', 'f']:
            with pytest.raises(Exception):
                drop_na(d, [col])
