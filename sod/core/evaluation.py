'''
Evaluation (classifiers creation and performance estimation)

Created on 1 Nov 2019

@author: riccardo
'''
import json
from multiprocessing import Pool, cpu_count
from os import makedirs
from os.path import abspath, join, dirname, isfile, isdir, basename, splitext
from itertools import repeat, product, chain, cycle
import warnings
from collections import defaultdict
from datetime import timedelta
import time

import click
import numpy as np
import pandas as pd
from pandas.core.indexes.range import RangeIndex
from joblib import dump, load
from sklearn.metrics.classification import (confusion_matrix,
                                            log_loss as scikit_log_loss)

from sod.core import pdconcat, odict
from sod.core.dataset import is_outlier, CLASSES, CLASSNAMES, ID_COL


CORRECTLY_PREDICTED_COL = 'correctly_predicted'
LOGLOSS_COL = 'log_loss'
OUTLIER_COL = 'outlier'
MODIFIED_COL = 'modified'
WINDOW_TYPE_COL = 'window_type'
UNIQUE_ID_COLUMNS = [ID_COL, OUTLIER_COL, MODIFIED_COL, WINDOW_TYPE_COL]


def drop_duplicates(dataframe, columns, decimals=0, verbose=True):
    '''Drops duplicates per class

    :return: a VIEW of `dataframe`. If you want
        to modify the returned dataframe safely (no pandas warnings), call
        `copy()` on it first.
    '''
    o_dataframe = dataframe
    dataframe = keep_cols(o_dataframe, columns).copy()

    class_index = np.zeros(len(o_dataframe))
    for i, selector in enumerate(CLASSES.values(), 1):
        class_index[selector(dataframe)] = i
    dataframe['_t'] = class_index

    if decimals > 0:
        for col in columns:
            dataframe[col] = np.around(dataframe[col], decimals)
    cols = list(columns) + ['_t']
    dataframe.drop_duplicates(subset=cols, inplace=True)
    if len(dataframe) == len(o_dataframe):
        if verbose:
            print('No duplicated row (per class) found')
        return o_dataframe
    dataframe = o_dataframe.loc[dataframe.index, :]
    if verbose:
        print('Duplicated (per class) found: %d rows removed' %
              (len(o_dataframe) - len(dataframe)))
        # print(dfinfo(dataframe))
    return dataframe


def drop_na(dataframe, columns, verbose=True):
    '''
    Remove rows with non finite values.under the specified columns
        (a value is finite when it is neither NaN nor +-Infinity)

    :return: a VIEW of dataframe with only rows with finite values.
        If you want to modify the returned DataFrame, call copy() on it
    '''
    tot = len(dataframe)
    with pd.option_context('mode.use_inf_as_na', True):
        dataframe = dataframe.dropna(axis=0, subset=list(columns), how='any')
    if dataframe.empty:
        raise ValueError('All values NA')
    elif verbose:
        if len(dataframe) == tot:
            print('No row removed (no NA found)')
        else:
            print('%d NA rows removed' % (tot - len(dataframe)))

    return dataframe


def keep_cols(dataframe, columns):
    '''
    Drops all columns of dataframe not in `columns` (preserving in any case
    the columns in `UNIQUE_ID_COLUMNS` found in dataframe's columns)

    :return: a VIEW of the original dataframe (does NOT return a copy)
        keeping the specified columns only
    '''
    dfcols = set(dataframe.columns)
    cols = chain((k for k in columns if k not in UNIQUE_ID_COLUMNS),
                 (k for k in UNIQUE_ID_COLUMNS if k in dfcols))
    return dataframe[list(cols)]


def classifier(clf_class, dataframe, **clf_params):
    '''Returns a OneClassSVM classifier fitted with the data of
    `dataframe`.

    :param dataframe: pandas DataFrame
    :param columns: list of string denoting the columns of `dataframe` that
        represents the feature space to fit the classifier with
    :param clf_params: parameters to be passed to `clf_class`
    '''
    clf = clf_class(**clf_params)
    clf.fit(dataframe.values)
    return clf


def predict(clf, dataframe, *columns):
    '''
    :return: a DataFrame with columns 'correctly_predicted' (boolean),
        'log_loss' (float) plus the columns of `dataframe` useful to uniquely
        identify the row: these columns depend on the dataset used and are
        `UNIQUE_ID_COLUMNS` (or a subset of it).
    '''

#     OneClassSVM.decision_function(self, X):
#         """Signed distance to the separating hyperplane.
#         Signed distance is positive for an inlier and negative for an outlier

#     IsolationForest.def decision_function(self, X):
#         ''' ...
#         Returns
#         -------
#         scores : array, shape (n_samples,)
#             The anomaly score of the input samples.
#             The lower, the more abnormal. Negative scores represent outliers,
#             positive scores represent inliers.
#
#         """

    predicted = _predict(
        clf,
        dataframe if not columns else dataframe[list(columns)]
    )
    outliers = is_outlier(dataframe)
    logloss_ = log_loss(outliers, predicted)
    cpred_ = correctly_predicted(outliers, predicted)
    dfcols = set(dataframe.columns)
    data = {
        CORRECTLY_PREDICTED_COL: cpred_,
        LOGLOSS_COL: logloss_,
        **{k: dataframe[k] for k in UNIQUE_ID_COLUMNS if k in dfcols}
    }
    return pd.DataFrame(data, index=dataframe.index)


def log_loss(outliers, predictions, eps=1e-15, normalize_=True):
    '''Computes the log loss of each prediction

    :param outliers: Ground truth (correct) labels (True: outlier,
        False: inlier)
    :param predictions: predicted scores (must have the same length as
        `outliers`) as returned by `_predict`: float array with positive
        (inlier) or negative (outlier) scores

    :return: a numpy array the same length of `predictions` with the log loss
        scores
    '''
    predictions_n = np.copy(predictions)
    if normalize_:
        # normalize bounds to have all predictions in [-1, 1]. But
        # normalize positives and then negatives separately, as we might have
        # unbalanced bounds (e.g. negatives in [-1, 0], positives in [0, 1200]
        positives = predictions_n > 0
        if np.nansum(positives):
            predictions_n[positives] = \
                predictions_n[positives] / np.nanmax(predictions_n[positives])
        negatives = predictions_n < 0
        if np.nansum(negatives):
            predictions_n[negatives] = \
                predictions_n[negatives] / -np.nanmin(predictions_n[negatives])

    not_outliers = ~outliers
    predictions_n[not_outliers] = (1 + predictions_n[not_outliers]) / 2.0
    predictions_n[outliers] = (1 - predictions_n[outliers]) / 2.0
    if eps is not None:
        predictions_n = np.clip(predictions_n, eps, 1 - eps)
    return -np.log10(predictions_n)  # http://wiki.fast.ai/index.php/Log_Loss


def correctly_predicted(outliers, predictions):
    '''Returns if the instances are correctly predicted

    :param outliers: boolean array denoting if the element is an outlier
    :param predictions: the output of `_predict`: float array with positive
        (inlier) or negative (outlier) scores

    :return: a boolean numpy array telling if the element is correctly
        predicted
    '''
    return (outliers & (predictions < 0)) | ((~outliers) & (predictions >= 0))


def _predict(clf, dataframe):
    '''Returns a numpy array of len(dataframe) integers where:
    negative values represent samples classified as OUTLIER (the lower, the
        more abnormal)
    positive values represent samples classified as INLIER (the higher, the
        more normal)
    The returned values bounds depend on the classifier chosen

    :param clf: the given (trained) classifier
    :param dataframe: pandas DataFrame
    :param columns: list of string denoting the columns of `dataframe` that
        represents the feature space to fit the classifier with
    '''
    # this method is very trivial it is used mainly for test purposes (mock)
    return clf.decision_function(dataframe.values)


# NOTE: IF YOU WANT TO CHANGE 'ok' or 'outlier' THEN CONSIDER CHANGING
# ALSO THE ROWS (SEE `_CLASSNAMES`). If you want to add new columns,
# also ADD a sort order in CMATRIX_SCORE_COLUMNS (see below)
CMATRIX_COLUMNS = ('ok', 'outlier', '% rec.', 'Mean %s' % LOGLOSS_COL)


def cmatrix_df(predicted_df):
    '''Returns a (custom) confusion matrix in the form of a dataframe
    The rows of the dataframe will be the keys of `CLASSES` (== `CLASSNAMES`),
    the columns 'ok' (inlier), 'outlier' '%rec', 'Mean log loss'.
    '''
    # NOTE: IF YOU WANT TO CHANGE 'ok' or 'outlier' THEN CONSIDER CHANGING
    # ALSO THE ROWS (SEE `_CLASSNAMES`)
    sum_df_cols = CMATRIX_COLUMNS
    sum_df = pd.DataFrame(index=CLASSNAMES,
                          data=[[0] * len(sum_df_cols)] * len(CLASSNAMES),
                          columns=sum_df_cols,
                          dtype=int)

    for typ, selectorfunc in CLASSES.items():
        cls_df = predicted_df[selectorfunc(predicted_df)]
        if cls_df.empty:
            continue
        correctly_pred = cls_df[CORRECTLY_PREDICTED_COL].sum()
        avg_log_loss = cls_df[LOGLOSS_COL].mean()
        # map any class defined here to the index of the column above which denotes
        # 'correctly classified'. Basically, map 'ok' to zero and any other class
        # to 1:
        col_idx = 0 if typ == CLASSNAMES[0] else 1
        # assign value and caluclate percentage recognition:
        sum_df.loc[typ, sum_df_cols[col_idx]] += correctly_pred
        sum_df.loc[typ, sum_df_cols[1-col_idx]] += len(cls_df) - correctly_pred
        sum_df.loc[typ, sum_df_cols[2]] = \
            np.around(100 * np.true_divide(correctly_pred, len(cls_df)), 3)
        sum_df.loc[typ, sum_df_cols[3]] = np.around(avg_log_loss, 5)

    return sum_df


def _get_eval_report_html_template():
    with open(join(dirname(__file__), 'eval_report_template.html'), 'r') as _:
        return _.read()


CLASSWEIGHTS = {
    CLASSNAMES[0]: 100,
    CLASSNAMES[1]: 100,
    CLASSNAMES[2]: 10,
    CLASSNAMES[3]: 50,
    CLASSNAMES[4]: 5,
    CLASSNAMES[5]: 1
}


def create_evel_report(cmatrix_dfs, outfilepath=None, title='%(title)s'):
    '''Saves the given confusion matrices dataframe to html

    :param cmatrix_df: a dict of string keys
        mapped to dataframe as returned from the function `cmatrix_df`
    :param outfilepath: the output file path. The extension will be
        appended as 'html', if an extension is not set
    :param title: the HTML title page
    '''
    # score columns: list of 'asc', 'desc' or None relative to each element of
    # CMATRIX_COLUMNS
    score_columns = {
        CMATRIX_COLUMNS[-2]: 'desc',  # '% rec.'
        CMATRIX_COLUMNS[-1]: 'asc'  # 'Mean %s' % LOGLOSS_COL
    }
    evl = [
        {'key': params, 'data': sumdf.values.tolist()}
        for (params, sumdf) in cmatrix_dfs.items()
    ]
    content = _get_eval_report_html_template() % {
        'title': title,
        'evaluations': json.dumps(evl),
        'columns': json.dumps(CMATRIX_COLUMNS),
        'scoreColumns': json.dumps(score_columns),
        'currentScoreColumn': json.dumps(CMATRIX_COLUMNS[-1]),
        'weights': json.dumps([CLASSWEIGHTS[_] for _ in CLASSNAMES]),
        'classes': json.dumps(CLASSNAMES)
    }
    if outfilepath is not None:
        if splitext(outfilepath)[1].lower() not in ('.htm', '.html'):
            outfilepath += ' .html'
        with open(outfilepath, 'w') as opn_:
            opn_.write(content)

    return content


def save_df(dataframe, filepath, **kwargs):
    '''Saves the given dataframe as HDF file under `filepath`.

    :param kwargs: additional arguments to be passed to pandas `to_df`,
        EXCEPT 'format' and 'mode' that are set inside this function
    '''
    dataframe.to_hdf(
        filepath,
        format='table', mode='w',
        **kwargs
    )


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
        if not len(train.index):
            raise ValueError('Training set empty during train_test_split')
        test = dataframe.loc[samples, :]
        if not len(test.index):
            raise ValueError('Test set empty during train_test_split')
        yield train, test


class EvalResult:

    __slots__ = ['clf', 'predictions', 'params', 'features']

    def __init__(self, clf, params, features):
        self.clf = clf
        self.params = params
        self.predictions = None
        self.features = features

    def predict(self, dataframe):
        self.predictions = None if dataframe.empty else \
            predict(self.clf, dataframe, *self.features)


def fit_and_predict(clf_class, train_df, columns, params, test_df=None,
                    filepath=None):
    '''Fits a model with `train_df[columns]` and returns an
    `EvalResult` with predictions derived from `test_df[columns]`

    :param params: dict of the classifier parameters. It can also be an
        iterable of (key, value) tuples or lists (which will be converted to
        the {key:value ... } corresponding dict)
    '''
    if filepath is not None and isfile(filepath):
        clf = load(filepath)
    else:
        clf = classifier(clf_class, train_df[list(columns)], **params)
    evres = EvalResult(clf, params, columns)
    if test_df is not None:
        evres.predict(test_df)
    if filepath is not None and not isfile(filepath):
        dump(clf, filepath)
    return evres


class CVEvaluator:
    '''Creates (and saves) a statisical ML model for outlier detection,
    and launches in parallel a CV
    evaluation, saving all predictions in HDF file, and a summary report in a
    dynamic html page'''

    # a dict of default params for the classifier:
    default_clf_params = {}

    def __init__(self, clf_class, parameters, n_folds=5):
        '''

        :param parameters: a dict mapping each parameter name (strings) to its
            list of possible values. The total number of cv iterations will
            be done for all possible combinations of all parameters values
        '''
        assert n_folds >= 1
        self.clf_class = clf_class
        self.n_folds = n_folds
        # setup self.parameters:
        __p = []
        for pname, vals in parameters.items():
            if not isinstance(vals, (list, tuple)):
                raise TypeError(("'%s' must be mapped to a list or tuple of "
                                 "values, even when there is only one value "
                                 "to iterate over") % str(pname))
            __p.append(tuple((pname, v) for v in vals))
        self.parameters = tuple(dict(_) for _ in product(*__p))
        self._predictions = defaultdict(list)
        self._eval_reports = defaultdict(dict)

    def uniquefilepath(self, destdir, *features, **params):
        '''Returns an unique file path from the given features and params'''
        # build an orderd dict
        paramz = odict()
        # add features:
        if features:
            paramz['features'] = features
        # add parameters:
        # 1. make params sorted first by iteration params then default ones,
        pr1 = sorted(k for k in params if k not in self.default_clf_params)
        pr2 = sorted(k for k in params if k in self.default_clf_params)
        # add them to paramz:
        for k in pr1 + pr2:
            paramz[k] = params[k]
        # build a base file path with the current classifier class name:
        basepath = join(destdir, self.clf_class.__name__)
        # add the URLquery-like string with ParamsEncDec.tostr:
        return basepath + ParamsEncDec.tostr(paramz)

    def run(self, dataframe, columns, remove_na, destdir):
        '''Runs the model evaluation using the data in `dataframe` under
        the specified columns and for all provided parameters
        '''
        basepath = self.uniquefilepath(destdir)
        print('Running CVEvaluator. All files will be stored in:\n%s' %
              dirname(basepath))
        print('with file names prefixed with "%s"' % basename(basepath))

        if remove_na:
            print('')
            dataframe = drop_na(dataframe,
                                set(_ for lst in columns for _ in lst),
                                verbose=True).copy()

        # first check: all dataframes are non-empty. This might be due to a set of
        # n-folds for which ...
        for _1, _2 in self.train_test_split_cv(dataframe):
            pass

        def cpy(dfr):
            return dfr if dfr is None else dfr.copy()

        self._predictions.clear()
        self._eval_reports.clear()
        pool = Pool(processes=int(cpu_count()))

        with click.progressbar(
            length=len(columns) * (1 + self.n_folds) * len(self.parameters),
            fill_char='o', empty_char='.'
        ) as pbar:

            def aasync_callback(result):
                self._applyasync_callback(pbar, result, destdir)

            def kill_pool(err_msg):
                print('ERROR:')
                print(err_msg)
                try:
                    pool.terminate()
                except ValueError:  # ignore ValueError('pool not running')
                    pass

            for cols in columns:
                # purge the dataframe from duplicates (drop_duplicates)
                # and unnecessary columns (keep_cols). Return a copy at the end
                # of the process. This helps memory mamagement in
                # sub-processes (especialy keep_cols + copy)
                dataframe_ = drop_duplicates(dataframe, cols, 0, verbose=False)
                dataframe_ = keep_cols(dataframe_, cols).copy()
                for params in self.parameters:
                    _traindf, _testdf = self.train_test_split_model(dataframe_)
                    prms = {**self.default_clf_params, **dict(params)}
                    fpath = \
                        self.uniquefilepath(destdir, *cols, **prms) + '.model'
                    pool.apply_async(
                        fit_and_predict,
                        (self.clf_class, cpy(_traindf), cols, prms,
                         cpy(_testdf), fpath),
                        callback=aasync_callback,
                        error_callback=kill_pool
                    )
                    for train_df, test_df in \
                            self.train_test_split_cv(dataframe_):
                        pool.apply_async(
                            fit_and_predict,
                            (self.clf_class, cpy(train_df), cols, prms,
                             cpy(test_df), None),
                            callback=aasync_callback,
                            error_callback=kill_pool
                        )

            pool.close()
            pool.join()

    def train_test_split_cv(self, dataframe):
        '''Returns an iterable yielding (train_df, test_df) elements for
        cross-validation. Both DataFrames in each yielded elements are subset
        of `dataframe`
        '''
        return train_test_split(dataframe, self.n_folds)

    def train_test_split_model(self, dataframe):  # pylint: disable=no-self-use
        '''Returns two dataframe representing the train and test dataframe for
        training the global model. Unless subclassed this method returns the
        tuple:
        ```
        dataframe, None
        ```
        '''
        return dataframe, None

    def _applyasync_callback(self, pbar, eval_result, destdir):
        pbar.update(1)

        # make three hashable and sortable objects using unique file path
        # (destdir = '' because it's useless)
        pkey = self.uniquefilepath(destdir, **eval_result.params)
        fkey = self.uniquefilepath(destdir, *eval_result.features)
        fpkey = self.uniquefilepath(destdir, *eval_result.features,
                                    **eval_result.params)

        self._predictions[fpkey].append(eval_result.predictions)
        # (eval_result.predictions might be None)

        if len(self._predictions[fpkey]) == self.n_folds + 1:
            # finished with predictions of current parameters for the current
            # features, save as hdf5 and store the summary dataframe of the
            # predicted segments just saved:
            self._eval_reports[fkey][pkey] = self.save_predictions(fpkey)

        sum_dfs = self._eval_reports.get(fkey, {})
        if 0 < len(sum_dfs) == len(self.parameters):
            # finished with the current eval report for the current features,
            # save as csv:
            self.save_evel_report(fkey)

    def save_predictions(self, fpkey, delete=True):
        '''saves the predictions dataframe to HDF and returns a cmatrix_df
        holding prediction's confusion matrix
        '''
        predicted_df = pdconcat(list(_ for _ in self._predictions[fpkey]
                                     if not getattr(_, 'empty', True)))
        # fpkey is also the unique file path associated to the data, so:
        save_df(predicted_df, fpkey + '.predictions.hdf', key='predictions')
        if delete:
            # delete unused data (help gc?):
            del self._predictions[fpkey]
        # now save the summary dataframe of the predicted segments just saved:
        return cmatrix_df(predicted_df)

    def save_evel_report(self, fkey, delete=True):
        features = ParamsEncDec.todict(fkey)['features'].replace(',', ', ')
        title = \
            self.clf_class.__name__ + " (features: %s)" % features
        sum_dfs = self._eval_reports[fkey]
        # Each sum_df key is the filename of the predictions_df.
        # Make that key readable by showing parameters only and separating them
        # via space (not '&'):
        sum_dfs = {
            basename(key)[basename(key).index('?')+1:].replace('&', ' '): val
            for key, val in sum_dfs.items()
        }
        # fkey is also the unique file path associated to the data
        create_evel_report(sum_dfs, fkey + '.evalreport.html', title)
        if delete:
            del self._eval_reports[fkey]


class ParamsEncDec:

    @staticmethod
    def _tostr(value):
        '''converts value to string. This is equal to `str(value)` with one
        exception: iterables neither strings nor bytes (lists, tuples) are
        returned sorting its elements and joining them with the comma,
        basically sorting them and returning the `str(value)` without square
        brackets. E.g.:
        [3, 1] = "1,3"
        ['a', 2, 3] = "2,3,a"
        '''
        if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
            try:
                value = sorted(value)
            except TypeError:
                # list/tuple with mixed types. Convert to string and then sort:
                value = sorted(str(_) for _ in value)
            return ",".join(value)
        return str(value)

    @staticmethod
    def tostr(params):
        '''Encodes params dict to string in a URL query fashion:
        ?param1=value&param2=value2...
        Each value is converted using `ParamsEncDec._tostr` (which is the
        same as Python __str__ method excepts that for iterables, where it
        firsts sort them and then returns the elements joined with comma)

        :param params: dict of string params mapped to values (pass OrederdDict
            if you want to preserve insertion order and Python < 3.7)
        '''
        chunks = []
        prefix = '?'
        for key, val in params.items():
            chunks.append('%s%s=%s' %
                          (prefix, str(key), ParamsEncDec._tostr(val)))
            prefix = '&'

        return ''.join(chunks)

    @staticmethod
    def todict(string, comma_sep=False):
        '''Decodes string encoded parameters to dict

        :param string: string including a query string denoting parameters
            used (encoded with `tostr` above). E.g., it can be
            a full file path:
            /user/me/file?a=8,9,10&b=gamma
            a file name:
            file?a=8,9,10&b=gamma
            or simply its query string portion:
            ?a=8,9,10&b=gamma
        :param comma_sep: boolean (False by default), whether strings with
            commas should be parsed back as tuples. False will leave every
            dict value as string
        '''
        ret = odict()  # ordered dict
        pth = splitext(basename(string))[0]
        if '?' not in pth:
            return ret
        pth = pth[pth.find('?') + 1:]
        splits = pth.split('&')
        for chunk in splits:
            param, value = chunk.split('=')
            if comma_sep and ',' in value:
                value = tuple(value.split(','))
            ret[param] = value
        return ret


class Evaluator:
    '''Class for evaluating a pre-fitted and saved model(s) (ususally obtained
    via `CVEvaluator`) against a
    dataset of instances, saving to file the predictions and an html report
    of the classifiers performances
    '''
    def __init__(self, classifier_paths, normalizer_df=None):
        if len(set(basename(_) for _ in classifier_paths)) != \
                len(classifier_paths):
            raise ValueError('You need to pass a list of unique file names')

        if not all('features' in ParamsEncDec.todict(_)
                   for _ in classifier_paths):
            raise ValueError("'features=' not found in all classifiers names")

        self.clfs = {}
        for _ in classifier_paths:
            try:
                if not isfile(_):
                    raise FileNotFoundError('File not found: "%s"' % _)
                self.clfs[basename(_)] = load(_)
            except Exception as exc:
                raise ValueError('Error reading "%s": %s' % (_, str(exc)))

        # if normalizer_df is provided get min and max for all features:
        bounds = None
        if normalizer_df is not None:
            bounds = {}
            for _ in classifier_paths:
                for feat in ParamsEncDec.todict(_)['features'].split(','):
                    bounds[feat] = (None, None)
            normalizer_df_columns = set(normalizer_df.columns)
            ndf = normalizer_df[~is_outlier(normalizer_df)]
            for feat in list(bounds.keys()):
                if feat not in normalizer_df_columns:
                    raise ValueError('"%s" not in normalizer dataframe' % feat)
                bounds[feat] = np.nanmin(ndf[feat]), np.nanmax(ndf[feat])
        self.bounds = bounds

    def run(self, test_df, destdir):
        pool = Pool(processes=int(cpu_count()))

        print('Running Evaluator (%d classifiers supplied)' % len(self.clfs))

        with click.progressbar(
            length=len(self.clfs),
            fill_char='o', empty_char='.'
        ) as pbar:

            def imap_func(arg):
                name, clf, dfr, feats = arg
                return name, predict(clf, dfr, *feats)

            def kill_pool(err_msg):
                print('ERROR:')
                print(err_msg)
                try:
                    pool.terminate()
                except ValueError:  # ignore ValueError('pool not running')
                    pass

            try:
                cmatrix_dfs = {}
                for clfname, predicted_df in \
                        pool.imap_unordered(imap_func, self.iterator(test_df)):
                    pbar.update(1)
                    if destdir is not None:
                        save_df(predicted_df,
                                join(destdir, clfname+'.predictions.hdf'),
                                key='predictions')
                    key = splitext(clfname)[0].replace('?', ' ').replace('&', ' ')
                    cmatrix_dfs['clf=%s' % key] = cmatrix_df(predicted_df)

                pool.close()
                pool.join()

            except Exception as exc:  # pylint: disable=broad-except
                kill_pool(exc)

            outfilepath = None if destdir is None else \
                join(destdir, 'evalreport.html')
            title = ('Evalation results comparing '
                     '%d classifiers') % len(self.clfs)
            return create_evel_report(cmatrix_dfs, outfilepath, title=title)

    def iterator(self, dataframe):
        for name, clf in self.clfs.items():
            features = ParamsEncDec.todict(name)['features'].split(',')
            dataframe_ = drop_duplicates(dataframe, features, 0, verbose=False)
            dataframe_ = keep_cols(dataframe_, features).copy()
            # normalize:
            if self.bounds is not None:
                for feat in features:
                    min_, max_ = self.bounds[feat]
                    dataframe_.loc[:, feat] = \
                        (dataframe_[feat] - min_) / (max_ - min_)

            yield name, clf, dataframe_, features




# def is_prediction_dataframe(dataframe):
#     '''Returns whether the given dataframe is the result of predictions
#     on a trained classifier
#     '''
#     return CORRECTLY_PREDICTED_COL in dataframe.columns


# def normalize_predictions(predictions, inbounds=None, outbounds=(-1, 1)):
#     '''Returns predictions (numpy array of values where negative
#     values represent OUTLIERS and positive ones INLIERS) normalized in
#     [outbounds[0], outbounds[1]]
#
#     :param inbounds: the input bounds, if None, it will be inferred from
#         `predictions` min and max (ignoring NaNs)
#     :param predictions: the output of `_predict`: float array with positive
#         (inlier) or negative (outlier) scores
#
#     '''
#     if inbounds is None:
#         imin, imax = np.nanmin(predictions), np.nanmax(predictions)
#     else:
#         imin, imax = inbounds
#     omin, omax = outbounds
#     # map predictions to -1 1 and then to 0 1
#     ret = omin + (omax - omin) * (predictions - imin) / (imax - imin)
#     if inbounds is not None:
#         ret[ret < -1] = -1
#         ret[ret > 1] = 1
#     return ret
