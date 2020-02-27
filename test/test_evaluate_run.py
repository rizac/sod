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
from sod.core import paths, pdconcat
from datetime import datetime
from sod.core.paths import EVALUATIONS_CONFIGS_DIR


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

    root = dirname(__file__)
    datadir = join(root, 'data')

    evalconfig = 'eval.allset_train_test.iforest.yaml'
#     clfevalconfig = join(root, 'data',
#                          'clfeval.allset_train_test.iforest.psd@5sec.yaml')

    if not isfile(join(datadir, 'allset_train.hdf_')):
        N = 200
        d = pd.DataFrame(
            {
                'Segment.db.id': list(range(N)),
                'dataset_id': 3,
                'channel_code': 'c',
                'location_code': 'l',
                'outlier': False,
                'hand_labelled': True,
                'window_type': True,
                'event_time': datetime.utcnow(),
                'station_id': 5,
                # add nan to the features to check we drop na:
                'psd@2sec': np.append([np.nan, np.nan], np.random.random(N-2)),
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

        d.loc[N2-5:N2+5, :].to_hdf(join(datadir, 'allset_test.hdf_'), 'a', mode='w',
                 format='table')
    tmpdir = join(dirname(__file__), 'tmp')


    # REMOVE THESE LINES (CREATE A SMALL DATASET FOR TESTING FROM AN EXISTING ONE):
#     hdf_ = pd.read_hdf('/Users/riccardo/work/gfz/projects/sources/python/sod/sod/datasets/allset_train.hdf')
#     hts = []
#     for sc in pd.unique(hdf_.subclass):
#         for clname in allset_train.classnames:
#             hts.append(hdf_[(hdf_.subclass == sc) & (hdf_.outlier)][:10])
#             hts.append(hdf_[(hdf_.subclass == sc) & ~(hdf_.outlier)][:10])
#     save_df(pdconcat(hts),
#             '/Users/riccardo/work/gfz/projects/sources/python/sod/test/data/allset_train.hdf_',
#             key='allset_train')

    # @patch('sod.core.dataset.dataset_path')
    @patch('sod.core.evaluation.Pool',
           side_effect=lambda *a, **v: PoolMocker())
    @patch('sod.evaluate.load_cfg')
    def test_evaluator(self,
                       # pytest fixutres:
                       # tmpdir
                       mock_load_cfg,
                       mock_pool,
                       #mock_dataset_in_path
                       ):
        if isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)
        makedirs(self.tmpdir)

        def load_cfg_side_effect(*a, **kw):
            dic = original_load_cfg(*a, **kw)
            dic['training']['input']['features'] = [['psd@5sec'], ['psd@2sec', 'psd@5sec']]
            dic['training']['classifier']['parameters']['n_estimators'] = [10, 20]
            dic['training']['classifier']['parameters']['max_samples'] = [5]
            dic['training']['input']['filename'] = 'allset_train.hdf_'
            dic['test']['filename'] = 'allset_test.hdf_'
            return dic

        mock_load_cfg.side_effect = load_cfg_side_effect

        with patch('sod.evaluate.DATASETS_DIR', self.datadir):
            with patch('sod.evaluate.EVALUATIONS_RESULTS_DIR', self.tmpdir):
    
                eval_cfg_path = self.evalconfig
                cvconfigname = basename(eval_cfg_path)
                evalsumpath = join(self.tmpdir,
                                   'summary_evaluationmetrics.hdf')

                runner = CliRunner()
                result = runner.invoke(run, ["-c", cvconfigname])
                assert not result.exception
                assert "4 of 4 models created " in result.output
                assert "4 of 4 predictions created " in result.output
                evalsum_df = pd.read_hdf(evalsumpath)
                assert len(evalsum_df) == 4
                mtime = stat(evalsumpath).st_mtime

                result = runner.invoke(run, ["-c", cvconfigname])
                assert not result.exception
                assert "0 of 4 models created " in result.output
                assert "0 of 4 predictions created " in result.output
                evalsum_df = pd.read_hdf(evalsumpath)
                assert len(evalsum_df) == 4
                assert stat(evalsumpath).st_mtime == mtime
