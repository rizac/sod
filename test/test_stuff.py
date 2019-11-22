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
from sod.core.plot import plot, plot_calibration_curve
from sod.core.dataset import open_dataset
from sod.core.evaluation import classifier


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
    def test_plot(self, mock_plt):
        plot(self.dfr, 'noise_psd@5sec', 'noise_psd@2sec', axis_lim=.945,
             clfs={'a': self.clf})
        plot_calibration_curve({'a': self.clf}, self.dfr,
                               ['noise_psd@5sec', 'noise_psd@2sec'])
        # plot_decision_func_2d(None, self.clf)
