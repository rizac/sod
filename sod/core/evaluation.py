'''
Evaluation (classifiers creation and performance estimation)

Created on 1 Nov 2019

@author: riccardo
'''
import json
import re
import importlib

from multiprocessing import Pool, cpu_count
from os import listdir, makedirs
from os.path import join, dirname, isfile, basename, splitext, isdir, abspath
from itertools import product, chain
from collections import defaultdict

from sklearn.ensemble.iforest import IsolationForest
from sklearn.svm.classes import OneClassSVM

from urllib.parse import quote as q, unquote as uq
import click
import numpy as np
import pandas as pd
from pandas.core.indexes.range import RangeIndex
from joblib import dump, load

from sod.core import (
    OUTLIER_COL, odict, CLASS_SELECTORS, CLASSNAMES, PREDICT_COL
)
from sod.core.metrics import log_loss, average_precision_score, roc_auc_score,\
    roc_curve, precision_recall_curve
# from sod.core.dataset import is_outlier, OUTLIER_COL


def classifier(clf_class, dataframe, **clf_params):
    '''Returns a scikit learn model fitted with the data of
    `dataframe`. If `destpath` is not None, saves the classifier to file

    :param dataframe: pandas DataFrame
    :param columns: list of string denoting the columns of `dataframe` that
        represents the feature space to fit the classifier with
    :param clf_params: dict of parameters to be passed to `clf_class`
    '''
    clf = clf_class(**clf_params)
    clf.fit(dataframe.values)
    return clf


def predict(clf, dataframe, features, columns=None):
    '''
    Builds and returns a PRDICTED DATAFRAME (predicted_df) with the predicted
    scores of the given classifier `clf` on the given `dataframe`.

    :param features: the columns of the dataframe to be used as features.

    :param columns: list/tuple/None. The columns of the dataframe to be kept
        and returned in the predicted dataframe together with the predicted
        scores. It is usually a subset of
        `dataframe` columns for memory performances. None (the default) or
        empty list/tuple: return all columns

    :return: a DataFrame with the columns 'outlier' (boolean) and
        'predicted_anomaly_score' (float in [0, 1]) plus the columns specified
        in the `columns` parameter
    '''
    predicted = _predict(
        clf,
        dataframe if not features else dataframe[list(features)]
    )
    cols = columns if columns else dataframe.columns.tolist()
    data = {
        **{c: dataframe[c] for c in cols},
        PREDICT_COL: predicted
    }
    return pd.DataFrame(data, index=dataframe.index)


def _predict(clf, dataframe):
    '''Returns a numpy array of len(dataframe) predictions, (i.e. floats
    in [0, 1]) for each row of dataframe.

    **A prediction is a float in [0, 1] representing the
    anomaly score (0: inlier, 1: outlier)**

    For any new classifier added, call the prediciton method which best fits
    your needs (e.g., predict, decision_function, score_samples) and, if
    needed, convert its output into a numpy array of numbers in [0, 1].
    Binary classifiers (e.g. SVMs) should return an array of numbers in
    EITHER 0 or 1 (this way some metrcis - e.g. log loss) are trivial, but we
    keep consistency).
    The currently implemented IsolationForest and OneClassSVM do so.

    :param clf: the given (trained) classifier
    :param dataframe: pandas DataFrame

    :return: a numopt array of numbers all in [0, 1]
    '''
    # this method is very trivial it is used mainly for test purposes (mock)
    if isinstance(clf, IsolationForest):
        return -clf.score_samples(dataframe.values)

    if isinstance(clf, OneClassSVM):
        # OCSVM.decision_function returns the Signed distance to the separating
        # hyperplane. Signed distance is positive for an inlier and negative
        # for an outlier.
        ret = clf.decision_function(dataframe.values)
        # OCSVMs do NOT support bounded scores, thus:
        ret[ret >= 0] = 0
        ret[ret < 0] = 1
        return ret

    raise ValueError('Classifier type not implemented in _predict: %s'
                     % str(clf))


def save_df(dataframe, filepath, **kwargs):
    '''Saves the given dataframe as HDF file under `filepath`.

    :param kwargs: additional arguments to be passed to pandas `to_df`,
        EXCEPT 'format' and 'mode' that are set inside this function
    '''
    if 'key' not in kwargs:
        key = splitext(basename(filepath))[0]
        if not re.match('^[a-zA-Z_]+$', key):
            raise ValueError('Invalid file basename. Provide a `key` argument '
                             'to the save_df function or change file name')
        kwargs['key'] = key
    kwargs.setdefault('append', False)
    dataframe.to_hdf(
        filepath,
        format='table',
        mode='w' if not kwargs['append'] else 'a',
        **kwargs
    )


def hdf_nrows(filepath):
    '''Gets the number of rows of the given HDF'''
    store = pd.HDFStore(filepath)
    try:
        keys = list(store.keys())
        if len(keys) != 1:
            raise ValueError('Unable to get nrows from testset: '
                             'HDF has more than 1 key')
        return store.get_storer(keys[0]).nrows
    finally:
        store.close()


def split(size, n_folds):
    '''Iterable yielding tuples of `(start, end)` indices
    (`start` < `end`) resulting from splitting `size` into `n_folds`.
    Both start and end are >= than 0 and <= `size`
    '''
    if size < n_folds:
        raise ValueError('Cannot split %s elements in %d folds' % (size,
                                                                   n_folds))
    if n_folds < 2:
        raise ValueError('Cannot split: folds <= 1')
    step = np.true_divide(size, n_folds)
    for i in range(n_folds):
        yield int(np.ceil(i*step)), int(np.ceil((i+1)*step))


def train_test_split(dataframe, n_folds=5):
    '''Iterable yielding tuples of `(train, test)` dataframes.
    Both `train` and `test` are disjoint subsets of `dataframe`, their union
    is equal to `dataframe`.
    `test` results from a random splitting of `dataframe` into `n_folds`
    subsets, `train` will be in turn composed of the rows of `dataframe` not
    in `test`
    '''
    if not isinstance(dataframe.index, RangeIndex):
        if not np.issubdtype(dataframe.index.dtype, np.integer) or \
                len(pd.unique(dataframe.index.values)) != len(dataframe):
            raise ValueError("The dataframe index must be composed of unique "
                             "numeric integer values")
    indices = np.copy(dataframe.index.values)
    last_iter = n_folds - 1
    for _, (start, end) in enumerate(split(len(dataframe), n_folds)):
        if _ < last_iter:
            samples = np.random.choice(indices, size=end-start,
                                       replace=False)
            indices = indices[np.isin(indices, samples, assume_unique=True,
                                      invert=True)]
        else:
            # avoid randomly sampling, we have just to use indices:
            samples = indices
        train = dataframe.loc[~dataframe.index.isin(samples), :]
        # dataframe.empty returns true also if no columns, so:
        if not len(train.index):  # pylint: disable=len-as-condition
            raise ValueError('Training set empty during train_test_split')
        test = dataframe.loc[samples, :]
        if not len(test.index):  # pylint: disable=len-as-condition
            raise ValueError('Test set empty during train_test_split')
        yield train, test


#############################
# evaluation cli functions: #
#############################


# for urllib.quote, make safe all characters except chars <= ' ' (32), chr(127)
# and /&?=,%
_safe = ''.join(set(chr(_) for _ in range(33, 127)) - set('/&?=,%'))


class TrainingParam:
    '''Class handling the parameter for creating N models
    and by means of the function `_classifier_mp`.
    This class is intended to be used inside `run_evaluation`
    from within a multiprocessing.Pool with several worker sub-processes to
    parallelize and speed up the calculations.
    The number N of models is inferred from the parameter passed in `__init__`.
    When calling `iterargs` with a given destination directory, this class
    returns a list of N arguments to be passed to `_classifier_mp`.
    See `run_evaluation` for details.
    Note: `_classifier_mp` is not implemented inside this class for
    pickable problem with multiprocessing
    '''
    def __init__(self, clf_classname, clf_param_dict,
                 trainingset_filepath, input_features_list,
                 trainingset_drop_na):
        try:
            modname = clf_classname[:clf_classname.rfind('.')]
            module = importlib.import_module(modname)
            classname = clf_classname[clf_classname.rfind('.')+1:]
            clf_class = getattr(module, classname)
            if clf_class is None:
                raise Exception('classifier class is None')
            self.clf_class = clf_class
        except Exception as exc:
            raise ValueError('Invalid `clf`: check paths and names.'
                             '\nError : %s' % str(exc))
        # setup self.parameters:
        __p = []
        for pname, vals in clf_param_dict.items():
            if not isinstance(vals, (list, tuple)):
                raise TypeError(("'%s' must be mapped to a list or tuple of "
                                 "values, even when there is only one value "
                                 "to iterate over") % str(pname))
            __p.append(tuple((pname, v) for v in vals))
        self.parameters = tuple(dict(_) for _ in product(*__p))

        if not isfile(trainingset_filepath):
            raise FileNotFoundError(trainingset_filepath)
        self.trainingset_filepath = trainingset_filepath
        self.features = input_features_list
        self.drop_na = trainingset_drop_na

    @property
    def allfeatures(self):
        ret = []
        for feats in self.features:
            for feat in feats:
                if feat not in ret:
                    ret.append(feat)
        return tuple(ret)

    def iterargs(self, destdir):
        '''Builds and returns from this object parameters a list of N arguments
        to be passed to `_create_save_classifier`. See `run_evaluation` for
        details
        '''
        training_df = self.read_trainingset()
        ret = []
        for features in self.features:
            for params in self.parameters:
                destpath = join(
                    destdir,
                    self.model_filename(self.clf_class,
                                        basename(self.trainingset_filepath),
                                        *features, **params)
                )
                ret.append((
                    self.clf_class,
                    training_df[features],
                    params,
                    destpath
                ))
        return ret

    def read_trainingset(self):
        ret = pd.read_hdf(self.trainingset_filepath,
                          columns=self.allfeatures)
        if self.drop_na:
            ret = ret.dropna(axis=0, how='any')

        return ret

    @staticmethod
    def model_filename(clf_class, tr_set, *features, **clf_params):
        '''converts the given argument to a model filename, with extension
        .sklmodel'''
        pars = odict()
        pars['clf'] = [str(clf_class.__name__)]
        pars['tr_set'] = [
            str(_) for _ in
            (tr_set if isinstance(tr_set, (list, tuple)) else [tr_set])
        ]
        pars['feats'] = [str(_) for _ in features]
        for key in sorted(clf_params):
            val = clf_params[key]
            pars[q(key, safe=_safe)] = [
                str(_) for _ in
                (val if isinstance(val, (list, tuple)) else [val])
            ]

        return '&'.join("%s=%s" % (k, ','.join(q(_, safe=_safe) for _ in v))
                        for k, v in pars.items()) + '.sklmodel'


def _classifier_mp(args):
    '''Creates and saves models from the given argument. See `TrainingParam`
    and `run_evaluation`'''
#     (clf_class, input_filepath, features, params, destpath, drop_na) = args
#     if not isdir(dirname(destpath)):
#         raise ValueError('Can not store model, parent directory does not exist: '
#                          '"%s"' % destpath)
#     elif isfile(destpath):
#         return destpath, False
#     dataframe = pd.read_hdf(input_filepath, columns=features)
#     if drop_na:
#         dataframe = dataframe.dropna(axis=0, subset=features, how='any')
#     clf = classifier(clf_class, dataframe, **params)
#     dump(clf, destpath)
#     return destpath, True

    (clf_class, training_df, params, destpath) = args
    if isfile(destpath):
        return None, destpath
    clf = classifier(clf_class, training_df, **params)
    # We could save here because although this function is executed from
    # a worker proces, there can not be conflicts. However, for safety,
    # delagate to the parent (main) process. Thus comment this:
    # dump(clf, destpath)
    return clf, destpath


DEF_CHUNKSIZE = 200000


class TestParam:
    '''Class handling the parameter(s) for testing N models against a test set
    and saving the predictions to HDF file by means of the function
    `_predict_mp`. This class is intended to be used inside `run_evaluation`
    from within a multiprocessing.Pool with several worker sub-processes to
    parallelize and speed up the calculations.
    The number N of models is passed in `set_classifiers_paths`.
    After that, one calls `iterargs` with a given test dataframe to yield
    N arguments to be passed to `_predict_mp`.
    Note that `iterargs` will be called several times with chunks
    of the test dataframe to predict. This is necessary to avoid problems
    for big test sets.
    See `run_evaluation` for details.
    Note: `_predict_mp` is not implemented inside this class for
    pickable problem with multiprocessing
    '''
    def __init__(self, testset_filepath, categorical_columns, columns2save,
                 drop_na):
        if not isfile(testset_filepath):
            raise FileNotFoundError(testset_filepath)
        self.testset_filepath = testset_filepath
        self.categorical_columns = categorical_columns
        self.columns2save = columns2save
        self.drop_na = drop_na
        self.classifiers_paths = {}

    def set_classifiers_paths(self, classifiers_paths):
        '''Builds and returns from this object parameters a list of N arguments
        to be passed to `_create_save_classifier`. See `run_evaluation` for
        details
        '''
        ret = odict()
        for clfpath in classifiers_paths:
            outdir = splitext(clfpath)[0]
            destpath = join(outdir, basename(self.testset_filepath))
            if isfile(destpath):
                continue
            elif not isdir(dirname(destpath)):
                makedirs(dirname(destpath))
            if not isdir(dirname(destpath)):
                continue
            feats = self.model_params(clfpath)['feats']
            ret[clfpath] = (destpath, feats)
        self.classifiers_paths = ret

    @property
    def num_iterations(self):
        '''Gets the number of iterations for creating the predicitons
        dataframes. This is the number of chunks to be read from the test set
        times the number of classifiers set with set_classifiers'''
        nrows = hdf_nrows(self.testset_filepath)
        chunksize = DEF_CHUNKSIZE
        return len(self.classifiers_paths) * \
            int(np.ceil(np.true_divide(nrows, chunksize)))

    def iterargs(self, test_df):
        '''Yields an iterable of arguments
        to be passed to `_create_save_classifier`. See `run_evaluation` for
        details
        '''
        for clfpath, (destpath, feats) in self.classifiers_paths.items():
            yield (test_df, clfpath, feats, self.columns2save,
                   self.drop_na, destpath)

    def read_testset(self):
        chunksize = DEF_CHUNKSIZE
        categ_columns = None  # to be initialized
        for dataframe in pd.read_hdf(self.testset_filepath,
                                     columns=self._allcolumns,
                                     chunksize=chunksize):
            if categ_columns is None:
                categ_columns = set(self.categorical_columns or []) & \
                    set(_ for _ in dataframe.columns)
            for c__ in categ_columns:
                dataframe[c__] = dataframe[c__].astype('category')
            yield dataframe

    @property
    def _allcolumns(self):
        allcols = list(self.columns2save)
        for clfpath in self.classifiers_paths:
            feats = self.model_params(clfpath)['feats']
            for feat in feats:
                if feat not in allcols:
                    allcols.append(feat)
        return allcols

    @staticmethod
    def model_params(model_filename):
        '''Converts the given model_filename (or absolute path) into a dict
        of key -> tuple of **strings** (single values parameters will be mapped
        to a 1 element tuple)
        '''
        pth = basename(model_filename)
        pth_, ext = splitext(pth)
        if ext == '.sklmodel':
            pth = pth_
        pars = pth.split('&')
        ret = odict()
        for par in pars:
            key, val = par.split('=')
            ret[uq(key)] = tuple(uq(_) for _ in val.split(','))
        return ret


def _predict_mp(args):
    '''Tests a given models against a given test set and saves the prediction
    result as HDF. See `TestParam` and `run_evaluation`'''
    (test_df, clfpath, features, columns2save, drop_na, destpath) = args
    if drop_na:
        test_df = test_df.dropna(axis=0, subset=features, how='any')
    if test_df.empty:
        return None, destpath
    pred_df = predict(load(clfpath), test_df, features, columns2save)
    return pred_df, destpath


# MODELDIRNAME = 'models'
# EVALREPORTDIRNAME = 'evalreports'
# PREDICTIONSDIRNAME = 'predictions'


def run_evaluation(training_param, testing_param, destdir):
    '''Runs the model evaluation
    '''
    if not isdir(destdir):
        raise NotADirectoryError(''"%s"'' % destdir)
    print('Running Evaluator. All files will be stored in:\n%s' %
          destdir)

    classifier_paths = []

    print('Step 1 of 2: Training (creating models)')
    pool = Pool(processes=int(cpu_count()))
    print('Reading Training file (HDF)')
    iterargs = training_param.iterargs(destdir)
    with click.progressbar(length=len(iterargs),
                           fill_char='o', empty_char='.') as pbar:
        try:
            newly_created_models = 0
            print('Building classifiers from parameters and training file')
            for clf, destpath in \
                    pool.imap_unordered(_classifier_mp, iterargs):
                # save on the main process (here):
                if clf is not None:
                    dump(clf, destpath)
                    newly_created_models += 1
                classifier_paths.append(destpath)
                pbar.update(1)
            # absolutely call these methods
            # although in impa and imap unordered
            # do make sense?
            pool.close()
            pool.join()
        except Exception as exc:
            _kill_pool(pool, str(exc))
            raise exc
    print("%d of %d models created (already existing were not overwritten)" %
          (newly_created_models, len(classifier_paths)))

    print('Step 2 of 2: Testing (creating prediction data frames)')
    testing_param.set_classifiers_paths(classifier_paths)
    pool = Pool(processes=int(cpu_count()))
    pred_filepaths = []
    with click.progressbar(length=testing_param.num_iterations,
                           fill_char='o', empty_char='.') as pbar:
        try:
            for test_df_chunk in testing_param.read_testset():
                iterargs = testing_param.iterargs(test_df_chunk)
                for pred_df, destpath in \
                        pool.imap_unordered(_predict_mp, iterargs):
                    if pred_df is not None:
                        # save on the main process (here):
                        save_df(pred_df, destpath, append=True)
                        if destpath not in pred_filepaths:
                            pred_filepaths.append(destpath)
                    pbar.update(1)
            # absolutely call these methods,
            # although in imap and imap unordered
            # do make sense?
            pool.close()
            pool.join()
        except Exception as exc:
            _kill_pool(pool, str(exc))
            raise exc
    print("%d of %d predictions created (already existing were not overwritten)" %
          (len(pred_filepaths), len(classifier_paths)))

    print()
    print('Creating evaluation metrics')
    create_summary_evaluationmetrics(destdir)
    print('DONE')


def _kill_pool(pool, err_msg):
    print('ERROR:')
    print(err_msg)
    try:
        pool.terminate()
    except ValueError:  # ignore ValueError('pool not running')
        pass


def create_summary_evaluationmetrics(destdir):
    '''Creates a new HDF file storing some metrics for all precidtions
    (HDF files) found inside `destdir` and subfolders

    :param destdir: a destination directory **whose FILE SUBTREE STRUCTURE
        MUST HAVE BEEN CREATED BY `run_evaluation` or (if calling from scrippt file)
        `evaluate.py`: a list of scikit model files with associated directories
        (the model name without the extension '.sklmodel') storing each
        prediction run on HDF datasets.
    `'''
    eval_df_path = join(destdir, 'summary_evaluationmetrics.hdf')

    cols = [
        'model',
        'test_set',
        'log_loss',
        'roc_auc_score',
        'average_precision_score',
        'best_th_roc_curve',
        'best_th_pr_curve'
    ]

    if isfile(eval_df_path):
        dfr = pd.read_hdf(eval_df_path)
    else:
        dfr = pd.DataFrame(columns=cols, data=[])

    # a set is faster than a dataframe for searching already processed
    # couples of (clf, testset_hdf):
    already_processed_tuples = \
        set((tuple(_) for _ in zip(dfr.model, dfr.test_set)))
    newrows = []
    for clfname in [] if not isdir(destdir) else listdir(destdir):
        clfdir, ext = splitext(clfname)
        if ext != '.sklmodel':
            continue
        clfdir = join(destdir, clfdir)
        for testname in [] if not isdir(clfdir) else listdir(clfdir):
            if (clfname, testname) in already_processed_tuples:
                continue
            predicted_df = pd.read_hdf(join(clfdir, testname),
                                       columns=[OUTLIER_COL, PREDICT_COL])

            newrows.append({
                cols[0]: clfname,
                cols[1]: testname,
                cols[2]: log_loss(predicted_df),
                cols[3]: roc_auc_score(predicted_df),
                cols[4]: average_precision_score(predicted_df),
                cols[5]: roc_curve(predicted_df)[-1],
                cols[6]: precision_recall_curve(predicted_df)[-1]
            })

    if newrows:
        pd_append = pd.DataFrame(columns=cols, data=newrows)
        if dfr.empty:
            dfr = pd_append
        else:
            dfr = dfr.append(pd_append,
                             ignore_index=True,
                             verify_integrity=False,
                             sort=False).reset_index(drop=True)
        save_df(dfr, eval_df_path)

    