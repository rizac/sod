'''
Created on 28 Oct 2019

@author: riccardo
'''
import click

from os.path import isabs, abspath, isdir, isfile, dirname, join, basename, splitext
from yaml import safe_load
from sod.evaluation import Evaluator, is_outlier, datasets
from sklearn.svm.classes import OneClassSVM
from sklearn.ensemble.iforest import IsolationForest


def load_cfg(fname):
    if not isabs(fname):
        fname = join(inputcfgpath(), fname)
    fname = abspath(fname)
    with open(fname) as stream:
        return safe_load(stream)


def inputcfgpath():
    return abspath(join(dirname(__file__), 'executions'))


def _rootpath():
    return abspath(join(dirname(__file__), '..', '..'))


def inputpath():
    return abspath(join(_rootpath(), 'tmp', 'datasets'))


def outputpath():
    return abspath(join(_rootpath(), 'tmp', 'evaluation-results'))


class OcsvmEvaluator(Evaluator):

    # A dict of default params for the classifier. In principle, put here what
    # should not be iterated over, but applied to any classifier during cv:
    default_clf_params = {'cache_size': 1500}

    def __init__(self, parameters, n_folds=5):
        Evaluator.__init__(self, OneClassSVM, parameters, n_folds)

    def train_test_split_cv(self, dataframe):
        '''Returns an iterable yielding (train_df, test_df) elements for
        cross-validation. Both DataFrames in each yielded elements are subset
        of `dataframe`
        '''
        return Evaluator.train_test_split_cv(
            self, dataframe[~is_outlier(dataframe)]
        )

    def train_test_split_model(self, dataframe):
        '''Returns two dataframe representing the train and test dataframe for
        training the global model. Unless subclassed this method returns the
        tuple:
        ```
        dataframe, None
        ```
        '''
        is_outl = is_outlier(dataframe)
        return dataframe[~is_outl], dataframe[is_outl]

    def run(self, dataframe, columns,  # pylint: disable=arguments-differ
            output):
        Evaluator.run(self, dataframe, columns, remove_na=True, output=output)


class IsolationForestEvaluator(OcsvmEvaluator):

    # A dict of default params for the classifier. In principle, put here what
    # should not be iterated over, but applied to any classifier during cv:
    default_clf_params = {'behaviour': 'new'}  # , 'contamination': 0}

    def __init__(self, parameters, n_folds=5):
        Evaluator.__init__(self, IsolationForest, parameters, n_folds)


EVALUATORS = {
    'OneClassSVM': OcsvmEvaluator,
    'IsolationForest': IsolationForestEvaluator  # IsolationForestEvaluator
}


@click.command()
@click.option('-c', '--config', help='configuration YAML file NAME', required=True)
def run(config):
    cfg_dict = load_cfg(config)
    open_dataset = getattr(datasets, splitext(cfg_dict['input'])[0])

    evaluator_class = EVALUATORS.get(cfg_dict['clf'], None)
    if evaluator_class is None:
        raise ValueError('%s in the config is invalid, please specify: %s' %
                         ('clf', str(" ".join(EVALUATORS.keys()))))

    infile = join(inputpath(), cfg_dict['input'])
    print('Reading from: %s' % str(infile))
    outdir = join(outputpath(), basename(config))
    print('Saving to: %s' % str(outdir))
    evl = evaluator_class(cfg_dict['parameters'], n_folds=5)
    evl.run(open_dataset(infile, normalize_=cfg_dict['input_normalize']),
            columns=cfg_dict['features'], output=outdir)
    return 0


if __name__ == '__main__':
    run()  # pylint: disable=no-value-for-parameter
