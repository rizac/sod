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
from stream2segment.cli import process as s2s_process
from sod.core import paths, pdconcat
import os
import yaml


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

#     evalconfig_cv_notest = join(dirname(__file__), 'data', 'eval.allset.iforest.5cv.notest.yaml')
#     evalconfig_cv_test = join(dirname(__file__), 'data', 'eval.allset.iforest.5cv.test.yaml')
#     evalconfig_nocv_notest = join(dirname(__file__), 'data', 'eval.allset.iforest.nocv.notest.yaml')
#     evalconfig_nocv_test = join(dirname(__file__), 'data', 'eval.allset.iforest.nocv.test.yaml')
    
#     evalconfig2 = join(dirname(__file__), 'data', 'eval.pgapgv.yaml')
#     cv_evalconfig3 = join(dirname(__file__), 'data', 'cv.allset_train.iforest.yaml')
#     clf_evalconfig1 = join(dirname(__file__), 'data', 'eval.allset_train.yaml')

    tmpdir = join(dirname(__file__), 'tmp', 's2s')
    assert isdir(dirname(tmpdir))
    if isdir(tmpdir):
        shutil.rmtree(tmpdir)
    os.mkdir(tmpdir)

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

    @pytest.mark.parametrize('jnconfig_db_key, confname', [
        ('dbpath_eu_new', 's2s_2019_03_at_rs5'),
        ('dbpath_me', 'me_at_rz_minus_vm258'),
        ('dbpath_chile', 'sod_chile_at_rs5'),
    ])
    def test_evaluator(self,
                       # pytest fixutres:
                       jnconfig_db_key, confname
                       ):

        INPATH = join(self.datadir, 's2s')
        assert isdir(INPATH)
        OUTPATH = self.tmpdir

        # get the configpath
        yamlfile = join(INPATH, confname+'.yaml')
        assert isfile(yamlfile)
        pyfiledir = join(dirname(dirname(__file__)), 'sod', 'stream2segment',
                         'configs')
        pyfile = join(pyfiledir, confname + '.py')
        assert isfile(pyfile)
        outfile = join(OUTPATH, confname + '.hdf')
        if isfile(outfile):
            os.remove(outfile)
        jupyterconffile = join(dirname(dirname(__file__)), "sod", "jupyter",
                               "jnconfig.yaml")
        assert isfile(jupyterconffile)
        with open(jupyterconffile) as _:
            dbdict = yaml.safe_load(_)
        dbpath = dbdict[jnconfig_db_key]

        assert not os.path.isfile(outfile)
        runner = CliRunner()
        result = runner.invoke(s2s_process, ["-c", yamlfile,
                                             "-p", pyfile,
                                             "-d", dbpath,
                                             "-mp",
                                             outfile])
        assert not result.exception
        assert os.path.isfile(outfile)
        dfr = pd.read_hdf(outfile)
        if jnconfig_db_key == 'dbpath_chile':
            assert sum(dfr.outlier) == 0
            assert sum(dfr.hand_labelled) == 2
            assert len(dfr) == 2
            assert sum(dfr.window_type) == len(dfr)
            assert pd.unique(dfr.dataset_id).tolist() == [3]
        elif jnconfig_db_key == 'dbpath_me':
            assert len(dfr[dfr.outlier & dfr.hand_labelled]) == 4
            assert len(dfr[(~dfr.outlier) & dfr.hand_labelled]) == 4
            assert len(dfr[(~dfr.outlier) & (~dfr.hand_labelled)]) == 4
            assert len(dfr) == 6 * 2  # *2 because window types noise and windows
            assert sum(dfr.window_type) == len(dfr)/2
            assert pd.unique(dfr.dataset_id).tolist() == [2]
        elif jnconfig_db_key == 'dbpath_eu_new':
            assert len(dfr[(~dfr.outlier) & (~dfr.hand_labelled)]) == 4
            assert len(dfr[(~dfr.outlier) & dfr.hand_labelled]) == 8
            assert len(dfr[dfr.outlier & dfr.hand_labelled]) == 4
            assert len(dfr) == 8 * 2  # *2 because window types noise and windows
            assert sum(dfr.window_type) == len(dfr)/2
            assert pd.unique(dfr.dataset_id).tolist() == [1]

            
                    

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
