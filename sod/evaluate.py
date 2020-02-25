'''
Created on 28 Oct 2019

@author: riccardo
'''
import importlib, sys

from os import makedirs
from os.path import (isabs, abspath, isdir, isfile, dirname, join, basename,
                     splitext, split)

from yaml import safe_load
import click

from sod.core.dataset import open_dataset
from sod.core.evaluation import Evaluator, ClfEvaluator
# from sklearn.svm.classes import OneClassSVM
# from sklearn.ensemble.iforest import IsolationForest
from sod.core.paths import EVALUATIONS_CONFIGS_DIR, EVALUATIONS_RESULTS_DIR
import traceback


def load_cfg(fname):
    if not isabs(fname):
        fname = join(EVALUATIONS_CONFIGS_DIR, fname)
    fname = abspath(fname)
    with open(fname) as stream:
        return safe_load(stream)


def _open_dataset(filepath, columns2save, categorical_columns=None):
    print()
    title = 'Opening %s' % filepath
    print(title)
    print('-' * len(title))
    dfr = open_dataset(filepath,
                       categorical_columns=categorical_columns,
                       verbose=True)
    _ = [_ for _ in columns2save if _ not in dfr.columns]
    if _:
        raise ValueError('These columns under the parameter '
                         '"columns2save" are not in the testset: '
                         '%s' % str(_))
    return dfr


@click.command()
@click.option(
    '-c', '--config',
    help='configuration YAML file name (in "sod/evaluations/configs")',
    required=True
)
def run(config):
    cfg_dict = load_cfg(config)

    try:
        columns2save = cfg_dict['columns2save']
    except KeyError:
        raise ValueError('Missing argument "columns2save" '
                         '(specify a list of stirings)')

    is_normal_eval = ('features' in cfg_dict) + ('parameters' in cfg_dict)
    if is_normal_eval == 2:
        print('Creating and optionally evaluating models from parameter sets')
    elif is_normal_eval == 0:
        print('Evaluating provided classifiers')
    else:
        raise ValueError('Config file does not seem neither a evaluation'
                         'nor an clf-evaluation config file')

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

    categorical_columns = cfg_dict.get('categorical_columns', None)
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
            test_df = _open_dataset(test_dataframe_path,
                                    columns2save,
                                    categorical_columns)

        train_df = _open_dataset(cfg_dict['trainingset'],
                                 columns2save,
                                 categorical_columns)

        try:
            evl = Evaluator(clf_class, parameters=cfg_dict['parameters'],
                            columns2save=columns2save,
                            cv_n_folds=cfg_dict['cv_n_folds'])
    
            # run(self, train_df, columns, destdir, test_df=None, remove_na=True):
            evl.run(train_df, columns=cfg_dict['features'], destdir=destdir,
                    test_df=test_df, remove_na=cfg_dict.get('remove_na', True))
        except Exception as exc:
            raise Exception(traceback.format_exc())
    else:
        try:
            classifier_paths = [abspath(join(EVALUATIONS_RESULTS_DIR, _))
                                for _ in cfg_dict['clf']]
    
            dataframe = _open_dataset(cfg_dict['testset'],
                                      columns2save,
                                      categorical_columns)
    
            evl = ClfEvaluator(classifier_paths,
                               columns2save=columns2save)
            evl.run(dataframe, cfg_dict['testset'], destdir)
        except Exception as exc:
            raise Exception(traceback.format_exc())

    return 0
    

if __name__ == '__main__':
    run()  # pylint: disable=no-value-for-parameter
