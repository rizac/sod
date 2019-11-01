'''
Created on 28 Oct 2019

@author: riccardo
'''
import click

from os.path import isabs, abspath, isdir, isfile, dirname, join, basename
from yaml import safe_load
from sod.evaluation import Evaluator, is_outlier, open_dataset
from sklearn.svm.classes import OneClassSVM


def load_cfg(fname):
    if not isabs(fname):
        fname = join(dirname(__file__), fname)
    fname = abspath(fname)
    with open(fname) as stream:
        params = safe_load(stream)

    if not isabs(params['input']):
        params['input'] = join(dirname(fname), params['input'])

    if not isabs(params['output']):
        params['output'] = join(dirname(fname), params['output'])

    return params


class OcsvmEvaluator(Evaluator):

    def __init__(self, parameters, rootoutdir=None, n_folds=5):
        Evaluator.__init__(self, OneClassSVM, parameters, rootoutdir, n_folds)

    def train_test_split_cv(self, dataframe):
        return Evaluator.train_test_split_cv(
            self, dataframe[~is_outlier(dataframe)]
        )

    def train_test_split_model(self, dataframe):
        is_outl = is_outlier(dataframe)
        return dataframe[~is_outl], dataframe[is_outl]

    def run(self, dataframe, columns):
        Evaluator.run(self, dataframe, columns, remove_na=True)


EVALUATORS = {
    'OneClassSVM': OcsvmEvaluator
}


@click.option('-c', '--config', help='configuration YAML file', required=True)
def run(config):
    cfg_dict = load_cfg(config)
    evaluator_class = EVALUATORS.get(cfg_dict['clf'], None)
    if evaluator_class is None:
        raise ValueError('%s in the config is invalid, please specify: %s' %
                         ('clf', str(" ".join(EVALUATORS.keys()))))
    
    evl = evaluator_class(cfg_dict['parameters'], cfg_dict['output'],
                          n_folds=5)
    evl.run(open_dataset(cfg_dict['input']), cfg_dict['features'])