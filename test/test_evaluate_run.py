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

from sod.core.evaluation import (split, classifier, predict, _predict,
                                 Evaluator, train_test_split,
                                 drop_duplicates,
                                 keep_cols, drop_na, cmatrix_df, ParamsEncDec,
                                 aggeval_hdf, aggeval_html, correctly_predicted,
    PREDICT_COL, save_df, log_loss, AGGEVAL_BASENAME)
from sod.core.dataset import (open_dataset, groupby_stations, allset,
    oneminutewindows, pgapgv)
from sod.evaluate import run
from sod.core import paths, pdconcat


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

    evalconfig_cv_notest = join(dirname(__file__), 'data', 'eval.allset.iforest.5cv.notest.yaml')
    evalconfig_cv_test = join(dirname(__file__), 'data', 'eval.allset.iforest.5cv.test.yaml')
    evalconfig_nocv_notest = join(dirname(__file__), 'data', 'eval.allset.iforest.nocv.notest.yaml')
    evalconfig_nocv_test = join(dirname(__file__), 'data', 'eval.allset.iforest.nocv.test.yaml')
    
#     evalconfig2 = join(dirname(__file__), 'data', 'eval.pgapgv.yaml')
#     cv_evalconfig3 = join(dirname(__file__), 'data', 'cv.allset_train.iforest.yaml')
    clf_evalconfig1 = join(dirname(__file__), 'data', 'eval.allset_train.yaml')

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
    def test_evaluator(self,
                       # pytest fixutres:
                       #tmpdir
                       mock_pool,
                       mock_dataset_in_path
                       ):
        if isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)
 
        INPATH, OUTPATH = self.datadir, self.tmpdir

        configs = [
            (self.evalconfig_nocv_notest, None, 0),
            (self.evalconfig_cv_test, self.clf_evalconfig1, 688),
            (self.evalconfig_nocv_test, self.clf_evalconfig1, 344),
            (self.evalconfig_cv_notest, self.clf_evalconfig1, 344)
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
                    
                    

#                 for (eval_cfg_path, clfeval_cfg_path) in configs:
#                     if clfeval_cfg_path is None:
#                         continue
# 
#                     runner = CliRunner()
#                     result = runner.invoke(run, ["-c", basename(eval_cfg_path)])
#                     # directory exists:
#                     assert result.exception
# 
#                     # now run evaluations with the generated files:
#                     runner = CliRunner()
#                     result = runner.invoke(run, ["-c", basename(clfeval_cfg_path)])
#                     assert not result.exception
# 
#         aggeval_html(join(OUTPATH, basename(self.cv_evalconfig2),
#                           'evalreports'))
#         aggeval_hdf(join(OUTPATH, basename(self.cv_evalconfig2),
#                          'evalreports'))
