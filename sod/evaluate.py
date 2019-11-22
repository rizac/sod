'''
Created on 28 Oct 2019

@author: riccardo
'''
import click

from os import makedirs
from os.path import (isabs, abspath, isdir, isfile, dirname, join, basename,
                     splitext)
from yaml import safe_load
from sod.core.dataset import dataset_path, open_dataset, magnitudeenergy
from sod.core.evaluation import CVEvaluator, is_outlier, Evaluator
from sklearn.svm.classes import OneClassSVM
from sklearn.ensemble.iforest import IsolationForest
from sod.core.paths import EVALUATIONS_CONFIGS_DIR, EVALUATIONS_RESULTS_DIR


def load_cfg(fname):
    if not isabs(fname):
        fname = join(EVALUATIONS_CONFIGS_DIR, fname)
    fname = abspath(fname)
    with open(fname) as stream:
        return safe_load(stream)


class OcsvmEvaluator(CVEvaluator):

    # A dict of default params for the classifier. In principle, put here what
    # should not be iterated over, but applied to any classifier during cv:
    default_clf_params = {'cache_size': 1500}

    def __init__(self, parameters, n_folds=5):
        CVEvaluator.__init__(self, OneClassSVM, parameters, n_folds)

    def train_test_split_cv(self, dataframe):
        '''Returns an iterable yielding (train_df, test_df) elements for
        cross-validation. Both DataFrames in each yielded elements are subset
        of `dataframe`
        '''
        return CVEvaluator.train_test_split_cv(
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
            destdir):
        CVEvaluator.run(self, dataframe, columns, remove_na=True,
                        destdir=destdir)


class IsolationForestEvaluator(OcsvmEvaluator):

    # A dict of default params for the classifier. In principle, put here what
    # should not be iterated over, but applied to any classifier during cv:
    default_clf_params = {'behaviour': 'new'}  # , 'contamination': 0}

    def __init__(self, parameters, n_folds=5):
        CVEvaluator.__init__(self, IsolationForest, parameters, n_folds)


class IsolationForestEvaluatorMagnitudeEnergyDataset(IsolationForestEvaluator):
    '''IsolationForest evaluator for the MagnitudeEnergy dataset'''

    @staticmethod
    def inlier_selector(dataframe):
        return ~is_outlier(dataframe) & \
            dataframe[magnitudeenergy._SUBCLASS_COL].str.match('^$')

    def train_test_split_cv(self, dataframe):
        '''Returns an iterable yielding (train_df, test_df) elements for
        cross-validation. Both DataFrames in each yielded elements are subset
        of `dataframe`
        '''
        # we have TWO types of not outliers in this
        return CVEvaluator.train_test_split_cv(
            self, dataframe[self.inlier_selector(dataframe)]
        )

    def train_test_split_model(self, dataframe):
        '''Returns two dataframe representing the train and test dataframe for
        training the global model. Unless subclassed this method returns the
        tuple:
        ```
        dataframe, None
        ```
        '''
        is_inl = self.inlier_selector(dataframe)
        return dataframe[is_inl], dataframe[~is_inl]

    def run(self, dataframe, columns,  # pylint: disable=arguments-differ
            destdir):
        CVEvaluator.run(self, dataframe, columns, remove_na=True,
                        destdir=destdir)


EVALUATORS = {
    'OneClassSVM': OcsvmEvaluator,
    'IsolationForest': IsolationForestEvaluator,  # IsolationForestEvaluator
    'IsolationForestMeDataset': IsolationForestEvaluatorMagnitudeEnergyDataset
}


@click.command()
@click.option(
    '-c', '--config',
    help='configuration YAML file name (in "sod/evaluations/configs")',
    required=True
)
def run(config):
    cfg_dict = load_cfg(config)

    is_cv_eval = ('features' in cfg_dict) + ('parameters' in cfg_dict)
    if is_cv_eval == 2:
        print('Running Cross validation evaluation and creating classifiers ')
    elif is_cv_eval == 0:
        print('Running evaluation of provided classifier path(s) ')
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

    if is_cv_eval:
        evaluator_class = EVALUATORS.get(cfg_dict['clf'], None)
        if evaluator_class is None:
            raise ValueError('%s in the config is invalid, please specify: %s'
                             % ('clf', str(" ".join(EVALUATORS.keys()))))

        dataframe = open_dataset(cfg_dict['input'],
                                 normalize=cfg_dict['input_normalize'])

        evl = evaluator_class(cfg_dict['parameters'], n_folds=5)
        evl.run(dataframe, columns=cfg_dict['features'], destdir=destdir)
    else:
        classifier_paths = [abspath(join(EVALUATIONS_RESULTS_DIR, _))
                            for _ in cfg_dict['clf']]

        dataframe = open_dataset(cfg_dict['input'], False)

        nrm_df = None
        if cfg_dict.get('input_normalize', None):
            nrm_df = open_dataset(cfg_dict['input_normalize'], False)

        evl = Evaluator(classifier_paths, normalizer_df=nrm_df)
        evl.run(dataframe, destdir)

    return 0


if __name__ == '__main__':
    run()  # pylint: disable=no-value-for-parameter
