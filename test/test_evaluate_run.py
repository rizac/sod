'''
Created on 11 Oct 2019

@author: riccardo
'''
import numpy as np
import pytest
from os import makedirs, listdir, stat
from os.path import join, abspath, dirname, isdir, isfile, basename, splitext
import pandas as pd
import shutil
from itertools import repeat, product
from collections import defaultdict
from sklearn.model_selection._split import KFold
from sklearn.metrics.classification import (confusion_matrix, brier_score_loss,
                                            log_loss as sk_log_loss)
import mock
from sklearn.svm.classes import OneClassSVM
from mock import patch
from click.testing import CliRunner
from sklearn.ensemble.iforest import IsolationForest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.scorer import brier_score_loss_scorer

# from sod.core.evaluation import (train_test_split,
#                                  drop_na, cmatrix_df, ParamsEncDec,
#                                  aggeval_hdf, aggeval_html, correctly_predicted,
#     PREDICT_COL, save_df, log_loss, AGGEVAL_BASENAME)
from sod.core.dataset import open_dataset
from sod.evaluate import run, load_cfg as original_load_cfg
from sod.core import paths, pdconcat, odict
from datetime import datetime
from sod.core.paths import EVALUATIONS_CONFIGS_DIR
from sod.core.evaluation import _predict_mp, classifier, load, dump, predict
import multiprocessing
import time

root = dirname(__file__)
datadir = join(root, 'data')
tmpdir = join(dirname(__file__), 'tmp')

if not isfile(join(datadir, 'allset_train.hdf_')):
    N = 210
    d = pd.DataFrame(
        {
            'Segment.db.id': list(range(N)),
            'dataset_id': 3,
            'channel_code': 'cha',
            'location_code': 'loc',
            'outlier': False,
            'hand_labelled': True,
            'window_type': True,
            'event_time': datetime.utcnow(),
            'station_id': 5,
            # add nan to the features to check we drop na:
            'psd@2sec': np.append(10*[np.nan], np.random.random(N-10)),
            'psd@5sec': np.random.random(N),
        }
    )
    # save with extension hdf_ BECAUSE hdf IS GITIGNORED!!!
    d.to_hdf(join(datadir, 'allset_train.hdf_'), 'a', mode='w',
             format='table')
    N2 = int(N/2)
    d['psd@2sec'] = np.append(np.random.random(N2), -np.random.random(N2))
    d['psd@5sec'] = np.append(np.random.random(N2), -np.random.random(N2))
    d['outlier'] = [True] * (N2-1) + [False, True] + [False] * (N2-1)

    dtest = d.loc[N2-5:N2+5, :]
    dtest = dtest.reset_index(drop=True).copy()
    # make first string shorter to test problems with min_itemsize:
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#string-columns
    dtest.loc[0, 'location_code'] = 'l'
    dtest.to_hdf(join(datadir, 'allset_test.hdf_'), 'a', mode='w',
                 format='table')


def _read_hdf(fpath):
    ret = []
    for i, _ in enumerate(pd.read_hdf(fpath, chunksize=10)):
        # time.sleep(np.random.randint(3))
        ret.append(_)
#         if i >= 10:
#             break
    return pd.concat(ret, axis=0, sort=False)


def tst_mp_hdf_read():
    '''this function was used to test that reading an HDF is not thread-safe
    (nor sub-process-safe): it seems that the handler used to open a file
    with pd.read_hdf(...chunksize=...) is global and thus every subprocess
    accesses the same opened file
    Now it is commented because its time consuming and its goal has alreayd
    been achieved
    '''
    fpath = join(datadir, 'allset_train.hdf_')
    p = multiprocessing.Pool()
    dfs = []
    for df in p.imap_unordered(_read_hdf, [fpath, fpath]):
        dfs.append(df)
    p.close()
    p.join()
    try:
        pd.testing.assert_frame_equal(dfs[0], dfs[1])
        raise ValueError('DataFrames are equal and should not')
    except AssertionError:
        pass

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

    evalconfig = 'eval.allset_train_test.iforest.00.yaml'
#     clfevalconfig = join(root, 'data',
#                          'clfeval.allset_train_test.iforest.psd@5sec.yaml')

    def tst_save_pred_df(self):
        input_filepath = join(datadir, 'allset_train.hdf_')
        dfr = pd.read_hdf(input_filepath)
        # use tmpdir because it will be removed (see test_evaluator below):
        clfpath = join(tmpdir, 'tmp.sklmodel')
        features = ['psd@2sec', 'psd@5sec']
        clf = classifier(IsolationForest,
                         dfr[features].dropna(subset=features))
        dump(clf, clfpath)
        categorical_columns = None
        columns2save = ['Segment.db.id']
        drop_na = True
        # use tmpdir because it will be removed (see test_evaluator below):
        outfile = join(tmpdir, 'tmp_pred.hdf') 
        args = (features, clfpath, input_filepath, categorical_columns, columns2save,
                drop_na, outfile)
        # use a chunksize of 10 because we have the first 10 numbers nan
        # in the test dataframe (see above) and we want to test that we
        # skip empty dataframes in _predict_and_save:
        with patch('sod.core.evaluation.DEF_CHUNKSIZE', 10):
            _predict_mp(args)
        pred_df1 = pd.read_hdf(outfile)
        pred_df2 = predict(clf, dfr[features + columns2save].dropna(subset=features),
                           features, columns2save)
        pd.testing.assert_frame_equal(pred_df1, pred_df2)


# TEST THE MAIN EVALUATION. FOR PROBLEMS, UNCOMMENT THE PATCH BELOW AND
# THE ARGUMENT mcok_pool. THIS WILL RUN THE TEST WITH A MOCKED VERSION OF
# multiprocessing.Pool WHICH EXECUTES EVERYTHING IN A SINGLE PROCESS

    @patch('sod.core.evaluation.Pool',
           side_effect=lambda *a, **v: PoolMocker())
    @patch('sod.evaluate.load_cfg')
    def test_evaluator(self,
                       mock_load_cfg,
                       mock_pool,
                       ):
        if isdir(tmpdir):
            shutil.rmtree(tmpdir)
        makedirs(tmpdir)

        def load_cfg_side_effect(*a, **kw):
            dic = original_load_cfg(*a, **kw)
            dic['training']['input']['features'] = [['psd@5sec'], ['psd@2sec', 'psd@5sec']]
            dic['training']['classifier']['parameters']['n_estimators'] = [10, 20]
            dic['training']['classifier']['parameters']['max_samples'] = [5]
            dic['training']['input']['filename'] = 'allset_train.hdf_'
            dic['test']['filename'] = 'allset_test.hdf_'
            return dic

        mock_load_cfg.side_effect = load_cfg_side_effect

        def read_all_pred_dfs():
            '''dict of pred_dataframe paths -> st_time'''
            ret = []
            for fle in listdir(tmpdir):
                _ = join(tmpdir, fle, 'allset_test.hdf_')
                if isfile(_):
                    ret.append(_)
            ret2 = odict()
            for _ in sorted(ret):
                ret2[_] = stat(_).st_mtime
            return ret2

        with patch('sod.evaluate.DATASETS_DIR', datadir):
            with patch('sod.evaluate.EVALUATIONS_RESULTS_DIR', tmpdir):
                with patch('sod.core.evaluation.DEF_CHUNKSIZE', 2):

                    eval_cfg_path = self.evalconfig
                    cvconfigname = basename(eval_cfg_path)
                    evalsumpath = join(tmpdir,
                                       'evaluationmetrics.hdf')

                    runner = CliRunner()
                    result = runner.invoke(run, ["-c", cvconfigname])
                    assert not result.exception
                    assert "4 of 4 models created " in result.output
                    assert "4 of 4 predictions created " in result.output
                    assert ("4 new prediction(s) found") in result.output
                    evalsum_df = pd.read_hdf(evalsumpath)
                    assert len(evalsum_df) == 4
                    mtime = stat(evalsumpath).st_mtime
                    pred_dfs = read_all_pred_dfs()

                    result = runner.invoke(run, ["-c", cvconfigname])
                    assert not result.exception
                    assert "0 of 4 models created " in result.output
                    assert "0 of 4 predictions created " in result.output
                    assert ("0 new prediction(s) found") in result.output
                    evalsum_df = pd.read_hdf(evalsumpath)
                    assert len(evalsum_df) == 4
                    assert stat(evalsumpath).st_mtime == mtime
                    pred_dfs2 = read_all_pred_dfs()
                    for _1, _2 in zip(pred_dfs.values(), pred_dfs2.values()):
                        assert _1 == _2

                    # now remove one evaluation summary:
                    evalsum_df = evalsum_df[:len(evalsum_df)-1]
                    evalsum_df.to_hdf(evalsumpath, key='a', format='table',
                                      mode='w')
                    assert len(evalsum_df) == 3
                    # wait one second because the mtime in mac is in second
                    # and we might have the same motification time even if the
                    # file has been modified
                    time.sleep(1)
                    # re-run:
                    result = runner.invoke(run, ["-c", cvconfigname])
                    assert not result.exception
                    assert "0 of 4 models created " in result.output
                    assert "0 of 4 predictions created " in result.output
                    assert ("1 new prediction(s) found") in result.output
                    evalsum_df = pd.read_hdf(evalsumpath)
                    assert len(evalsum_df) == 4
                    assert stat(evalsumpath).st_mtime > mtime
                    # and relaunch:
                    
