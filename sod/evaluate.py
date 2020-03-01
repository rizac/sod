'''
Created on 28 Oct 2019

@author: riccardo
'''
import importlib, sys

from os import makedirs
from os.path import (
    isabs, abspath, isdir, isfile, dirname, join, basename,
    splitext
)

from yaml import safe_load
import click

from sod.core.evaluation import TrainingParam, TestParam, run_evaluation
# from sklearn.svm.classes import OneClassSVM
# from sklearn.ensemble.iforest import IsolationForest
from sod.core.paths import EVALUATIONS_CONFIGS_DIR, EVALUATIONS_RESULTS_DIR,\
    DATASETS_DIR


def load_cfg(fname):
    if not isabs(fname):
        fname = join(EVALUATIONS_CONFIGS_DIR, fname)
    fname = abspath(fname)
    with open(fname) as stream:
        return safe_load(stream)


@click.command()
@click.option(
    '-c', '--config',
    help='configuration YAML file name (in "sod/evaluations/configs")',
    required=True
)
def run(config):
    cfg_dict = load_cfg(config)
    try:
        trn, tst = cfg_dict['training'], cfg_dict['test']

        training_param = TrainingParam(
            trn['classifier']['classname'],
            trn['classifier']['parameters'],
            join(DATASETS_DIR, trn['input']['filename']),
            trn['input']['features'],
            cfg_dict['drop_na'])

        test_param = TestParam(join(DATASETS_DIR, tst['filename']),
                               tst['save_options']['columns'],
                               cfg_dict['drop_na'],
                               tst['save_options']['min_itemsize'])
        run_evaluation(training_param, test_param,
                       EVALUATIONS_RESULTS_DIR)
    except KeyError as kerr:
        raise ValueError('Key not implemented in config.: "%s"' % str(kerr))
    return 0

if __name__ == '__main__':
    run()  # pylint: disable=no-value-for-parameter
