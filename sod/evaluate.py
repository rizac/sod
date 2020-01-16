'''
Created on 28 Oct 2019

@author: riccardo
'''
import importlib

from os import makedirs
from os.path import (isabs, abspath, isdir, isfile, dirname, join, basename,
                     splitext)

from yaml import safe_load
import click

from sod.core.dataset import (dataset_path, open_dataset, magnitudeenergy,
                              globalset)
from sod.core.evaluation import Evaluator, ClfEvaluator, is_outlier
from sklearn.svm.classes import OneClassSVM
from sklearn.ensemble.iforest import IsolationForest
from sod.core.paths import EVALUATIONS_CONFIGS_DIR, EVALUATIONS_RESULTS_DIR


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

    is_normal_eval = ('features' in cfg_dict) + ('parameters' in cfg_dict)
    if is_normal_eval == 2:
        print('Creating and optionally evaluating models from parameter sets')
    elif is_normal_eval == 0:
        print('Evaluating provided classifiers')
    else:
        raise ValueError('Config file does not seem neither a cv-evaluation'
                         'nor an evaluation config file')

    # get destdir:
    destdir = abspath(join(EVALUATIONS_RESULTS_DIR, basename(config)))
    if not isdir(destdir):
        makedirs(destdir)
        if not isdir(destdir):
            raise ValueError('Could not create %s' % destdir)
    elif isdir(destdir):
        raise ValueError("Output directory exists:\n"
                         "'%s'\n"
                         "Rename yaml file or remove directory" % destdir)
    print('Saving results (HDF, HTML files) to: %s' % str(destdir))

    if is_normal_eval:

        clf_class = None
        try:
            modname = cfg_dict['clf'][:cfg_dict['clf'].rfind('.')]
            module = importlib.import_module(modname)
            classname = cfg_dict['clf'][cfg_dict['clf'].rfind('.')+1:]
            clf_class = getattr(module, classname)
            if clf_class is None:
                raise Exception('classifier class is None')
        except Exception as exc:
            raise ValueError('Invalid `clf`: check paths and names.'
                             '\nError : %s' % str(exc))

        print('Using classifier: %s' % clf_class.__name__)
        test_df = None
        test_dataframe_path = cfg_dict['testset']
        if test_dataframe_path:
            if cfg_dict['input_normalize']:
                raise ValueError('No `test` allowed with `input_normalize`='
                                 'true. Set the `test` to "" (or null), or '
                                 'normalize the test dataframe first, and '
                                 'then run the evaluation with '
                                 '`input_normalize`=False')
            test_df = None  # open_dataset(test_dataframe_path, normalize=False)

        train_df = open_dataset(cfg_dict['trainingset'],
                                normalize=cfg_dict['input_normalize'])

        evl = Evaluator(clf_class, cfg_dict['parameters'],
                        cfg_dict['cv_n_folds'])
        # run(self, train_df, columns, destdir, test_df=None, remove_na=True):
        evl.run(train_df, columns=cfg_dict['features'], destdir=destdir,
                test_df=test_df, remove_na=cfg_dict.get('remove_na', True))
    else:
        classifier_paths = [abspath(join(EVALUATIONS_RESULTS_DIR, _))
                            for _ in cfg_dict['clf']]

        dataframe = open_dataset(cfg_dict['input'], False)

        nrm_df = None
        if cfg_dict.get('input_normalize', None):
            nrm_df = open_dataset(cfg_dict['input_normalize'], False)

        evl = ClfEvaluator(classifier_paths, normalizer_df=nrm_df)
        evl.run(dataframe, destdir)

    return 0


if __name__ == '__main__':
    run()  # pylint: disable=no-value-for-parameter
