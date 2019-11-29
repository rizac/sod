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

import mock
from sklearn.svm.classes import OneClassSVM
from mock import patch
from click.testing import CliRunner
from sklearn.ensemble.iforest import IsolationForest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.scorer import brier_score_loss_scorer
from sod.core.plot import plot, plot_calibration_curve, plotdist
from sod.core.dataset import open_dataset
from sod.core.evaluation import classifier, normalize


class Tester:

    datadir = join(dirname(__file__), 'data')

    with patch('sod.core.dataset.DATASETS_DIR', datadir):
        dfr = open_dataset(join(datadir, 'pgapgv.hdf_'), False)
        dfr2 = open_dataset(join(datadir, 'oneminutewindows.hdf_'), False)

    clf = classifier(OneClassSVM, dfr.iloc[:5,:][['delta_pga', 'delta_pgv']])

    evalconfig = join(dirname(__file__), 'data', 'pgapgv.ocsvm.yaml')
    evalconfig2 = join(dirname(__file__), 'data', 'oneminutewindows.ocsvm.yaml')

    tmpdir = join(dirname(__file__), 'tmp')

    @patch('sod.core.plot.plt')        
    def tst_plot(self, mock_plt):
        plot(self.dfr, 'noise_psd@5sec', 'noise_psd@2sec', axis_lim=.945,
             clfs={'a': self.clf})
        plot_calibration_curve({'a': self.clf}, self.dfr,
                               ['noise_psd@5sec', 'noise_psd@2sec'])
        plotdist(self.dfr)
        # plot_decision_func_2d(None, self.clf)

    def test_normalize(self):
        preds = [-1, 0, 1]
        assert (normalize(preds, range_in=None, map_to=[0, 1]) == [0, 0.5, 1]).all()
        assert (normalize(preds, range_in=None, map_to=[1, 0]) == [1, 0.5, 0]).all()
        assert (normalize(preds, range_in=None, map_to=[-1, 1]) == [-1, 0, 1]).all()
        assert (normalize(preds, range_in=None, map_to=[-3, -1]) == [-3, -2, -1]).all()
        assert (normalize(preds, range_in=None, map_to=[3, 1]) == [3, 2, 1]).all()
        
        preds = [-10, 0, 1]
        assert (normalize(preds, range_in=[-1, 1], map_to=[0, 1]) == [0, 0.5, 1]).all()
        assert (normalize(preds, range_in=[-1, 1], map_to=[1, 0]) == [1, 0.5, 0]).all()
        assert (normalize(preds, range_in=[-1, 1], map_to=[-1, 1]) == [-1, 0, 1]).all()
        assert (normalize(preds, range_in=[-1, 1], map_to=[-3, -1]) == [-3,-2, -1]).all()
        assert (normalize(preds, range_in=[-1, 1], map_to=[3, 1]) == [3, 2, 1]).all()
        
        preds = [-0.1, 0, 11.31]
        assert (normalize(preds, range_in=[-0.01, 1], map_to=[0, 1]) == [0, 0.5, 1]).all()
        assert (normalize(preds, range_in=[-0.01, 1], map_to=[1, 0]) == [1, 0.5, 0]).all()
        assert (normalize(preds, range_in=[-0.01, 1], map_to=[-1, 1]) == [-1, 0, 1]).all()
        assert (normalize(preds, range_in=[-0.01, 1], map_to=[-3, -1]) == [-3, -2, -1]).all()
        assert (normalize(preds, range_in=[-0.01, 1], map_to=[3, 1]) == [3, 2, 1]).all()
        
        preds = (-0.35693947774039203, 0.07255932122838979, 0.10972665377052404,
                 -0.3460104025237925, -0.2998335126365414, 0.09612788974091535,
                 -0.016918, 0)
        vals = normalize(preds, map_to=(0, 1))
        asd = 9
        
        