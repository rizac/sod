'''
Evaluation (classifiers creation and performance estimation)

Created on 1 Nov 2019

@author: riccardo
'''
import re
import importlib

from multiprocessing import Pool, cpu_count
from os import listdir, makedirs
from os.path import join, dirname, isfile, basename, splitext, isdir
from itertools import product

from sklearn.ensemble.iforest import IsolationForest
from sklearn.svm.classes import OneClassSVM
from urllib.parse import quote as q, unquote as uq
import click
import numpy as np
import pandas as pd
from pandas.core.indexes.range import RangeIndex
from joblib import dump, load

from sod.core import OUTLIER_COL, odict, PREDICT_COL, pdconcat
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
        if not re.match('^[a-zA-Z_][a-zA-Z0-9_]*$', key):
            raise ValueError('Invalid file basename. '
                             'Change the name or provide a `key` argument '
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

    def classifier_paths(self, destdir):
        return [_[0] for _ in self._clfiter(destdir)]

    def iterargs(self, destdir):
        '''Yields a list of N arguments
        to be passed to `_create_save_classifier`.
        Skips classifier(s) whose path already exist.
        See `run_evaluation` for details.
        **NOTE** This method reads the entire training set once
        before starting yielding, **be careful for performance reasons**.
        E.g., calling `args = list(...iterargs(...))` might take a while
        '''
        training_df = self.read_trainingset()
        for destpath, features, params in \
                self._clfiter(destdir):
            if isfile(destpath):
                continue
            yield (
                    self.clf_class,
                    training_df[features],
                    params,
                    destpath
                )

    def _clfiter(self, destdir):
        for features in self.features:
            for params in self.parameters:
                destpath = join(
                    destdir,
                    self.model_filename(self.clf_class,
                                        basename(self.trainingset_filepath),
                                        *features, **params)
                )
                yield destpath, features, params

    def read_trainingset(self):
        ret = pd.read_hdf(self.trainingset_filepath,
                          columns=self.allfeatures)
        if self.drop_na:
            ret = ret.dropna(axis=0, how='any')

        return ret

    @property
    def allfeatures(self):
        ret = []
        for feats in self.features:
            for feat in feats:
                if feat not in ret:
                    ret.append(feat)
        return tuple(ret)

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
        pars['feats'] = [str(_) for _ in sorted(features)]
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
    (clf_class, training_df, params, destpath) = args
#     if isfile(destpath):
#         return None, destpath
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
    def __init__(self, testset_filepath, columns2save,
                 drop_na, min_itemsize=None):
        if not isfile(testset_filepath):
            raise FileNotFoundError(testset_filepath)
        self.testset_filepath = testset_filepath
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#string-columns:
        self.min_itemsize = min_itemsize
        self.columns2save = columns2save
        self.drop_na = drop_na
        self.classifiers_paths = {}

    def set_classifiers_paths(self, classifiers_paths):
        '''Builds and returns from this object parameters a list of N arguments
        to be passed to `_create_save_classifier`.
        Skips classifiers whose prediction dataframe already exists. 
        See `run_evaluation` for details
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
        for dataframe in pd.read_hdf(self.testset_filepath,
                                     columns=self._allcolumns,
                                     chunksize=chunksize):
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
        to a 1-element tuple)
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


def run_evaluation(training_param, test_param, destdir):
    '''Runs the model evaluation
    '''
    if not isdir(destdir):
        raise NotADirectoryError(''"%s"'' % destdir)
    print('Running Evaluator. All files will be stored in:\n%s' %
          destdir)

    print('Step 1 of 2: Training (creating models)')
    newly_created_models = 0
    print('Reading Training file (HDF)')
    # returns iterable of classifier TO BE CREATED (already existing are not
    # yielded):
    iterargs = list(training_param.iterargs(destdir))
    if iterargs:
        pool = Pool(processes=int(cpu_count()))
        with click.progressbar(length=len(iterargs),
                               fill_char='o', empty_char='.') as pbar:
            try:
                print('Building classifiers from parameters and training file')
                for clf, destpath in \
                        pool.imap_unordered(_classifier_mp, iterargs):
                    # save on the main process (here):
                    if clf is not None:
                        dump(clf, destpath)
                        newly_created_models += 1
                    pbar.update(1)
                # absolutely call these methods
                # although in impa and imap unordered
                # do make sense?
                pool.close()
                pool.join()
            except Exception as exc:
                _kill_pool(pool, str(exc))
                raise exc

    classifier_paths = [
        _ for _ in training_param.classifier_paths(destdir) if isfile(_)
    ]
    print("%d of %d models created (already existing were not overwritten)" %
          (newly_created_models, len(classifier_paths)))

    print('Step 2 of 2: Testing (creating prediction data frames)')
    pred_filepaths = []
    test_param.set_classifiers_paths(classifier_paths)
    # with set_classifiers_paths above, we internally stored classifiers
    # who do NOT have a prediction dataframe. `num_iterations` below accounts
    # for thius, thus being zero for several reasons, among which also
    # if all classifiers already have a relative prediction dataframe:
    num_iterations = test_param.num_iterations
    if num_iterations:
        pool = Pool(processes=int(cpu_count()))
        with click.progressbar(length=num_iterations,
                               fill_char='o', empty_char='.') as pbar:
            try:
                for test_df_chunk in test_param.read_testset():
                    iterargs = test_param.iterargs(test_df_chunk)
                    for pred_df, destpath in \
                            pool.imap_unordered(_predict_mp, iterargs):
                        if pred_df is not None:
                            # save on the main process (here):
                            kwargs = {'append': True}
                            if test_param.min_itemsize:
                                kwargs['min_itemsize'] = test_param.min_itemsize
                            save_df(pred_df, destpath, **kwargs)
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
    print("%d of %d prediction HDF file(s) created "
          "(already existing were not overwritten)" %
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
    print('Computing summary evaluation metrics from '
          'predictions data frames (HDF file)')

    eval_df_path = join(destdir, 'evaluationmetrics.hdf')

    dfr, already_processed_tuples = None, set()
    if isfile(eval_df_path):
        dfr = pd.read_hdf(eval_df_path)
        already_processed_tuples = set(dfr._key)

    def build_key(clfdir, testname):
        return join(basename(clfdir), testname)

    # a set is faster than a dataframe for searching already processed
    # couples of (clf, testset_hdf):
    newrows = []
    clfs_prediction_paths = []
    for clfname in [] if not isdir(destdir) else listdir(destdir):
        clfdir, ext = splitext(clfname)
        if ext != '.sklmodel':
            continue
        clfdir = join(destdir, clfdir)
        for testname in [] if not isdir(clfdir) else listdir(clfdir):
            if build_key(clfdir, testname) in already_processed_tuples:
                continue
            clfs_prediction_paths.append((clfdir, testname))

    print('%d new prediction(s) found' % len(clfs_prediction_paths))
    if clfs_prediction_paths:
        errors = []
        pool = Pool(processes=int(cpu_count()))
        with click.progressbar(length=len(clfs_prediction_paths),
                               fill_char='o', empty_char='.') as pbar:
            for clfdir, testname, dic in \
                    pool.imap_unordered(_get_summary_evaluationmetrics_mp,
                                        clfs_prediction_paths):
                pbar.update(1)
                if isinstance(dic, Exception):
                    errors.append(dic)
                else:
                    dic['_key'] = build_key(clfdir, testname)
                    newrows.append(dic)

        if newrows:
            new_df = pd.DataFrame(data=newrows)
            if dfr is None:
                dfr = new_df
            else:
                dfr = pd.concat([dfr, new_df], axis=0, copy=True,
                                sort=False, ignore_index=True)
#                 dfr = dfr.append(new_df,
#                                  ignore_index=True,
#                                  verify_integrity=False,
#                                  sort=False).reset_index(drop=True)
            save_df(dfr, eval_df_path)

        if errors:
            print('%d prediction(s) discarded due to error' % len(errors))
            print('(possible cause: only one class found in the prediction)')


# columns denoting the metrics. Their value should be implemented in
# the next function
_METRIC_COLUMNS = (
    'log_loss',
    'roc_auc_score',
    'average_precision_score',
    'best_th_roc_curve',
    'best_th_pr_curve'
)


def _get_summary_evaluationmetrics_mp(clfdir_and_testname):
    clfdir, testname = clfdir_and_testname
    filepath = join(clfdir, testname)
    predicted_df = pd.read_hdf(filepath, columns=[OUTLIER_COL, PREDICT_COL])
    cols = tuple(_METRIC_COLUMNS)
    try:
        # parse the clf file name (which is the directory name of the
        # prediction dataframe we want to calculate metrics from), and
        # try to guess if they can be floats or ints:
        ret = odict()
        for key, value in TestParam.model_params(clfdir).items():
            # value is a tuple. First thing is to define how to store it
            # in a pandas DataFrame. Use its string method without brackets
            # (so that e.g., ['a', 'b'] will be stored as 'a,b' and
            # ['abc'] will be stored as 'abc':
            stored_value = ",".join(str(_) for _ in value)
            if len(value) == 1:
                try:
                    stored_value = int(value[0])
                except (ValueError, TypeError):
                    try:
                        stored_value = float(value[0])
                    except (ValueError, TypeError):
                        pass
            ret[key] = stored_value

        ret[cols[0]] = log_loss(predicted_df)
        ret[cols[1]] = roc_auc_score(predicted_df)
        ret[cols[2]] = average_precision_score(predicted_df)
        ret[cols[3]] = roc_curve(predicted_df)[-1]
        ret[cols[4]] = precision_recall_curve(predicted_df)[-1]

        return clfdir, testname, ret

    except Exception as exc:
        return clfdir, testname, exc
