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
                                            log_loss as sk_log_loss)
import mock
from sklearn.svm.classes import OneClassSVM
from mock import patch
from click.testing import CliRunner
from sklearn.ensemble.iforest import IsolationForest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.scorer import brier_score_loss_scorer

from sod.core.evaluation import (Evaluator, train_test_split,
                                 drop_na, cmatrix_df, ParamsEncDec,
                                 aggeval_hdf, aggeval_html, correctly_predicted,
    PREDICT_COL, save_df, log_loss, AGGEVAL_BASENAME)
from sod.core.dataset import open_dataset
from sod.evaluate import run, load_cfg as original_load_cfg
from sod.core import paths, pdconcat
from datetime import datetime


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

    evalconfig = join(root, 'data',
                      'eval.allset_train_test.iforest.yaml')
    clfevalconfig = join(root, 'data',
                         'clfeval.allset_train_test.iforest.psd@5sec.yaml')
    
    
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

    @patch('sod.core.dataset.dataset_path')
    @patch('sod.core.evaluation.Pool',
           side_effect=lambda *a, **v: PoolMocker())
    @patch('sod.evaluate.load_cfg')
    def test_evaluator(self,
                       # pytest fixutres:
                       # tmpdir
                       mock_load_cfg,
                       mock_pool,
                       mock_dataset_in_path
                       ):
        if isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

        def load_cfg_side_effect(*a, **kw):
            dic = original_load_cfg(*a, **kw)
            dic['features'] = [['psd@5sec'], ['psd@2sec', 'psd@5sec']]
            dic['parameters']['n_estimators'] = [10, 20]
            dic['parameters']['max_samples'] = [5]
            dic['trainingset'] = 'allset_train.hdf_'
            dic['testset'] = 'allset_test.hdf_'
            return dic

        mock_load_cfg.side_effect = load_cfg_side_effect

        INPATH, OUTPATH = self.datadir, self.tmpdir

        configs = [
            # eval_cfg_path, clfeval_cfg_path, expected_evaluated_instances:
            (self.evalconfig, None, 0)
        ]

        with patch('sod.evaluate.EVALUATIONS_CONFIGS_DIR', INPATH):
            with patch('sod.evaluate.EVALUATIONS_RESULTS_DIR', OUTPATH):

                mock_dataset_in_path.side_effect = \
                    lambda filename, *a, **v: join(INPATH, filename)

                for (eval_cfg_path, clfeval_cfg_path,
                     expected_evaluated_instances) in configs:
                    cvconfigname = basename(eval_cfg_path)
                    runner = CliRunner()
                    result = runner.invoke(run, ["-c", cvconfigname])
                    assert not result.exception

                    # check directory is created:
                    assert listdir(join(OUTPATH, cvconfigname))
                    # check subdirs are created:
                    subdirs = (Evaluator.EVALREPORTDIRNAME,
                               Evaluator.PREDICTIONSDIRNAME,
                               Evaluator.MODELDIRNAME)
                    assert sorted(listdir(join(OUTPATH, cvconfigname))) == \
                        sorted(subdirs)
                    # check for files in subdirs, but wait: if no cv
                    # and no test set, check only the model subdir:
                    if expected_evaluated_instances < 1:
                        # no saved file, check only for model files saved:
                        subdirs = [Evaluator.MODELDIRNAME]

                    # CHECK FOR FILES CREATED:
                    for subdir in subdirs:
                        filez = listdir(join(OUTPATH, cvconfigname, subdir))
                        assert filez
                        if subdir == Evaluator.EVALREPORTDIRNAME:
                            assert ('%s.html' % AGGEVAL_BASENAME) in filez
                            assert ('%s.hdf' % AGGEVAL_BASENAME) in filez

                    if expected_evaluated_instances < 1:
                        continue

                    # CHECK AND INSPECT PREDICTION DATAFRAMES:
                    prediction_file = \
                        listdir(join(OUTPATH, cvconfigname,
                                     Evaluator.PREDICTIONSDIRNAME))[0]
                    prediction_file = join(OUTPATH, cvconfigname,
                                           Evaluator.PREDICTIONSDIRNAME,
                                           prediction_file)
                    prediction_df = pd.read_hdf(prediction_file)
                    assert len(prediction_df) == expected_evaluated_instances

                    cols = allset.uid_columns
                    cols = sorted(list(cols) + [PREDICT_COL])
                    assert sorted(prediction_df.columns) == cols

