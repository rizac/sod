'''
Evaluation (classifiers creation and performance estimation)

Created on 1 Nov 2019

@author: riccardo
'''
import json
from multiprocessing import Pool, cpu_count
from os import listdir, makedirs
from os.path import join, dirname, isfile, basename, splitext, isdir, abspath
from itertools import product, chain
from collections import defaultdict

import click
import numpy as np
import pandas as pd
from pandas.core.indexes.range import RangeIndex
from joblib import dump, load
from sklearn.metrics.classification import (confusion_matrix,
                                            log_loss as scikit_log_loss)

from sod.core import pdconcat, odict
from sod.core.dataset import (is_outlier, OUTLIER_COL, dataset_info)
import re


CORRECTLY_PREDICTED_COL = 'correctly_predicted'
LOGLOSS_COL = 'log_loss'
PREDICT_COL = 'decision_function'


def drop_duplicates(dataframe, columns, decimals=0, verbose=True):
    '''Drops duplicates per class

    :return: a VIEW of `dataframe`. If you want
        to modify the returned dataframe safely (no pandas warnings), call
        `copy()` on it first.
    '''
    dinfo = dataset_info(dataframe)
    o_dataframe = dataframe
    dataframe = keep_cols(o_dataframe, columns).copy()

    class_index = np.zeros(len(o_dataframe))
    for i, cname in enumerate(dinfo.classnames, 1):
        selector = dinfo.class_selector[cname]
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
    dinfo = dataset_info(dataframe)
    cols = set(columns) | set(dinfo.uid_columns)
    # return the dataframe with only `cols` columns, but assure that
    # dinfo.uid_columns[0] is in the first position otherwise
    # dataset_info called on the returned dataframe won't work
    return dataframe[[dinfo.uid_columns[0]] +
                     list(_ for _ in cols if _ != dinfo.uid_columns[0])]


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
    dinfo = dataset_info(dataframe)
    # return the dataframe with only `cols` columns, but assure that
    # dinfo.uid_columns[0] is in the first position otherwise
    # dataset_info called on the returned dataframe won't work
    data = {
        dinfo.uid_columns[0]: dataframe[dinfo.uid_columns[0]],
        CORRECTLY_PREDICTED_COL: cpred_,
        LOGLOSS_COL: logloss_,
        PREDICT_COL: predicted,
        **{k: dataframe[k] for i, k in enumerate(dinfo.uid_columns) if i > 0}
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


def normalize(predictions, range_in=None, map_to=(0, 1)):
    '''Normalizes `predictions` and returns an array with values all within
    `maps_to`, where:
    the closer a value to `maps_to[0]`, the more it is an outlier
    the closer a value to `maps_to[1]`, the more it is an inlier

    :param predictions: the result of `predict`. Negative values must denote
        outliers, positive values must denote inliers
    :param range_in: 2-element tuple denoting the input range in the form:
        ```
        (outliers min, inliers max)
        ```
        The first element must be negative and the second positive.
        If None (the default), it defaults to the minimum of the negative
        scores of predictions, and the maximum of the positive scores of
        predictions
    :param map_to: 2-tuple element denoting the the output range
        in the form
        ```
        (value_of the_highest_outlier_score, value_of the_highest_inlier_score)
        ```
        The numbers DO NOT AHVE any restrictions:
        map_to=(1, 0) returns numbers in [0, 1], the higher, the more outlier
        map_to=(0, 1) returns numbers in [0, 1], the higher, the more inlier

    :return: new array with float values in map_to
    '''
    preds = np.array(predictions, copy=True, dtype=float)
    positives = preds >= 0
    if np.nansum(positives):
        if range_in is None:
            max_inlier = np.nanmax(preds[positives])
        else:
            max_inlier = range_in[1]
            if not max_inlier > 0:
                raise ValueError('range_in second element must be positive')
            preds[positives] = np.clip(preds[positives], None, max_inlier)
        preds[positives] = preds[positives] / max_inlier

    negatives = ~positives
    if np.nansum(negatives):
        if range_in is None:
            min_outlier = np.nanmin(preds[negatives])
        else:
            min_outlier = range_in[0]
            if not min_outlier < 0:
                raise ValueError('range_in first element must be negative')
            preds[negatives] = np.clip(preds[negatives], min_outlier, None)
        preds[negatives] = preds[negatives] / -min_outlier

    # we now have values all in [-1, 1], where values >=0 denote inliers
    if map_to is not None and (map_to[0] != -1 or map_to[1] != 1):
        preds = map_to[0] + (map_to[1] - map_to[0]) * (preds + 1.) / 2.0
    return preds


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
    dinfo = dataset_info(predicted_df)
    classnames = dinfo.classnames
    # NOTE: IF YOU WANT TO CHANGE 'ok' or 'outlier' THEN CONSIDER CHANGING
    # ALSO THE ROWS (SEE `CLASSNAMES`)
    sum_df_cols = CMATRIX_COLUMNS
    sum_df = pd.DataFrame(index=classnames,
                          data=[[0] * len(sum_df_cols)] * len(classnames),
                          columns=sum_df_cols,
                          dtype=int)

    for cname in classnames:
        cls_df = predicted_df[dinfo.class_selector[cname](predicted_df)]
        if cls_df.empty:
            continue
        correctly_pred = cls_df[CORRECTLY_PREDICTED_COL].sum()
        avg_log_loss = cls_df[LOGLOSS_COL].mean()
        # map any class defined here to the index of the column above which denotes
        # 'correctly classified'. Basically, map 'ok' to zero and any other class
        # to 1:
        col_idx = 1 if cls_df.iloc[0][OUTLIER_COL] else 0
        # assign value and caluclate percentage recognition:
        sum_df.loc[cname, sum_df_cols[col_idx]] += correctly_pred
        sum_df.loc[cname, sum_df_cols[1-col_idx]] += \
            len(cls_df) - correctly_pred
        sum_df.loc[cname, sum_df_cols[2]] = \
            np.around(100 * np.true_divide(correctly_pred, len(cls_df)), 3)
        sum_df.loc[cname, sum_df_cols[3]] = np.around(avg_log_loss, 5)

    sum_df.dataset_info = dinfo
    return sum_df


def _get_eval_report_html_template():
    with open(join(dirname(__file__), 'eval_report_template.html'), 'r') as _:
        return _.read()


def create_evel_report(cmatrix_dfs, outfilepath=None, title='%(title)s'):
    '''Creates and optionally saves the given confusion matrices dataframe
    to html

    :param cmatrix_df: a dict of string keys
        mapped to dataframe as returned from the function `cmatrix_df`
    :param outfilepath: the output file path. The extension will be
        appended as 'html', if an extension is not set. If None, nothing is
        saved
    :param title: the HTML title page

    :return: the HTML formatted string containing the evaluation report
    '''
    first_cm = next(iter(cmatrix_dfs.values()))
    dinfo = first_cm.dataset_info
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
    classnames = dinfo.classnames
    content = _get_eval_report_html_template() % {
        'title': title,
        'evaluations': json.dumps(evl),
        'columns': json.dumps(CMATRIX_COLUMNS),
        'scoreColumns': json.dumps(score_columns),
        'currentScoreColumn': json.dumps(CMATRIX_COLUMNS[-1]),
        'weights': json.dumps([dinfo.class_weight[_] for _ in classnames]),
        'classes': json.dumps(classnames)
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
        if not len(train.index):  # pylint: disable=len-as-condition
            raise ValueError('Training set empty during train_test_split')
        test = dataframe.loc[samples, :]
        if not len(test.index):  # pylint: disable=len-as-condition
            raise ValueError('Test set empty during train_test_split')
        yield train, test


class CVEvaluator:
    '''Creates (and saves) a statisical ML model for outlier detection,
    and launches in parallel a CV
    evaluation, saving all predictions in HDF file, and a summary report in a
    dynamic html page'''

    MODELDIRNAME = 'models'
    EVALREPORTDIRNAME = 'evalreports'
    PREDICTIONSDIRNAME = 'predictions'

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
        return basepath + ParamsEncDec.tostr(*features, **paramz)

    def run(self, dataframe, columns, remove_na, destdir):
        '''Runs the model evaluation using the data in `dataframe` under
        the specified columns and for all provided parameters
        '''
        basepath = self.uniquefilepath(destdir)
        print('Running CVEvaluator. All files will be stored in:\n%s' %
              dirname(basepath))
        print('with file names prefixed with "%s"' % basename(basepath))

        for subdir in [self.MODELDIRNAME, self.EVALREPORTDIRNAME,
                       self.PREDICTIONSDIRNAME]:
            _ddir = abspath(join(destdir, subdir))
            if not isdir(_ddir):
                makedirs(_ddir)
            if not isdir(_ddir):
                raise ValueError('Unable to create "%s"' % _ddir)

        if remove_na:
            print('')
            dataframe = drop_na(dataframe,
                                set(_ for lst in columns for _ in lst),
                                verbose=True).copy()

        def cpy(dfr):
            return dfr if dfr is None else dfr.copy()

        self._predictions.clear()
        self._eval_reports.clear()
        pool = None  # Pool(processes=int(cpu_count()))

        with click.progressbar(
            length=len(columns) * (1 + self.n_folds) * len(self.parameters),
            fill_char='o', empty_char='.'
        ) as pbar:

            def aasync_callback(result):
                pbar.update(1)
                clf, params, features, predictions = result
                self._applyasync_callback(clf, params, features, predictions,
                                          destdir)

            def kill_pool(err_msg):
                print('ERROR:')
                print(err_msg)
                try:
                    pool.terminate()
                except ValueError:  # ignore ValueError('pool not running')
                    pass

            try:
                for cols in columns:

                    # purge the dataframe from duplicates (drop_duplicates)
                    # and unnecessary columns (keep_cols). Return a copy at the end
                    # of the process. This helps memory mamagement in
                    # sub-processes (especialy keep_cols + copy)
                    dataframe_ = drop_duplicates(dataframe, cols, 0,
                                                 verbose=False)
                    dataframe_ = keep_cols(dataframe_, cols).copy()
                    for params in self.parameters:
                        pool = Pool(processes=int(cpu_count()))
                        _traindf, _testdf = \
                            self.train_test_split_model(dataframe_)
                        prms = {**self.default_clf_params, **dict(params)}
                        fpath = self.uniquefilepath(destdir, *cols, **prms)
                        fpath = join(dirname(fpath), self.MODELDIRNAME,
                                     basename(fpath)) + '.model'
                        pool.apply_async(
                            _fit_and_predict,
                            (self.clf_class, cpy(_traindf), cols, prms,
                             cpy(_testdf), fpath),
                            callback=aasync_callback,
                            error_callback=kill_pool
                        )
                        for train_df, test_df in \
                                self.train_test_split_cv(_traindf):
                            pool.apply_async(
                                _fit_and_predict,
                                (self.clf_class, cpy(train_df), cols, prms,
                                 cpy(test_df), None),
                                callback=aasync_callback,
                                error_callback=kill_pool
                            )

                        pool.close()
                        pool.join()

            except Exception as exc:  # pylint: disable=broad-except
                kill_pool(str(exc))

    def train_test_split_cv(self, dataframe):
        '''Split `dataframe` into random train and test subsets, yielding
        the tuple: ```(train_dataframe, test_dataframe)``` `n_folds` times

        :param dataframe: pandas DataFrame. It is the FIRST element returned by
            `train_test_split_model` (by default, the source dataframe
            unfiltered)
        '''
        return train_test_split(dataframe, self.n_folds)

    def train_test_split_model(self, dataframe):  # pylint: disable=no-self-use
        '''Returns two dataframe representing the train and test dataframe for
        training the global model. Unless subclassed this method returns the
        tuple:
        ```
        dataframe, None
        ```
        The first dataframe will be used to build the global model (saved as
        file) and tested against the second dataframe (if the latter is not
        None). Afterwards, it will be passed to `train_test_split_cv`
        '''
        return dataframe, None

    def _applyasync_callback(self, clf, params, features, predictions,
                             destdir):
        '''Callback executed from the apply_async above'''
        # make three hashable and sortable objects using unique file path
        # (destdir = '' because it's useless)
        pkey = self.uniquefilepath(destdir, **params)
        fkey = self.uniquefilepath(destdir, *features)
        fpkey = self.uniquefilepath(destdir, *features, **params)

        self._predictions[fpkey].append(predictions)
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
        destfile = join(dirname(fpkey), self.PREDICTIONSDIRNAME,
                        basename(fpkey)) + '.hdf'
        save_df(predicted_df, destfile, key='predictions')
        if delete:
            # delete unused data (help gc?):
            del self._predictions[fpkey]
        # now save the summary dataframe of the predicted segments just saved:
        return cmatrix_df(predicted_df)

    def save_evel_report(self, fkey, delete=True):
        features = ParamsEncDec.todict(fkey)['features']
        title = \
            self.clf_class.__name__ + " (features: %s)" % ", ".join(features)
        sum_dfs = self._eval_reports[fkey]
        # Each sum_df key is the filename of the predictions_df.
        # Make that key readable by showing parameters only and separating them
        # via space (not '&'):
        sum_dfs = {
            basename(key)[basename(key).index('?')+1:].replace('&', ' '): val
            for key, val in sum_dfs.items()
        }
        destfile = join(dirname(fkey), self.EVALREPORTDIRNAME,
                        basename(fkey)) + '.html'
        # fkey is also the unique file path associated to the data
        create_evel_report(sum_dfs, destfile, title)
        if delete:
            del self._eval_reports[fkey]


def _fit_and_predict(clf_class, train_df, columns, params, test_df=None,
                     filepath=None):
    '''Fits a model with `train_df[columns]` and returns the tuple
    clf, params, columns, predictions. Called from within apply_async
    in CVEvaluator

    :param params: dict of the classifier parameters
    '''
    if filepath is not None and isfile(filepath):
        clf = load(filepath)
    else:
        clf = classifier(clf_class, train_df[list(columns)], **params)
    predictions = None
    if test_df is not None and not test_df.empty:
        # evres.predict(test_df)
        predictions = predict(clf, test_df, *columns)
    if filepath is not None and not isfile(filepath):
        dump(clf, filepath)
    return clf, params, columns, predictions


def aggeval(indir, format='html', save=True):  # @ReservedAssignment
    '''Aggregates all evaluations html files of `indir` (produced with
    CVEvaluation) into a general evaluation in the specified format
    (html or hdf)
    '''
    if format == 'html':
        return aggeval_html(indir, save)
    if format == 'hdf':
        return aggeval_hdf(indir, save)
    raise ValueError('format can be either "hdf" or "html", not "%s"'
                     % str(format))


def aggeval_hdf(indir, save=True):
    '''Asuming `indir` is the directory where CV reports in HTML format
    have been saved by a run of `CVEvaluator`, then generates a new HTML
    page summing up all cv evaluation reports of `indir`
    '''
    all_data = []
    for (html_pre, data_list, classes, weights, columns, html_post) in \
            _evalhtmlscanner(indir):
        for dic in data_list:
            key = dic['key']
            newdic = {}
            allkeys = key.split(' ')
            for chunk in allkeys:
                key, val = chunk.split('=')
                try:
                    val = int(val)
                except:  # @IgnorePep8 pylint: disable=bare-except
                    try:
                        val = float(val)
                    except:  # @IgnorePep8 pylint: disable=bare-except
                        pass
                newdic[key] = val
            data = dic['data']
            # each data form column
            for classname, rowdata in zip(classes, data):
                new_df_row = dict(newdic)
                new_df_row['classname'] = classname
                for capt, val in zip(columns[2:], rowdata[2:]):
                    new_df_row[capt] = val
                all_data.append(new_df_row)

    dfr = pd.DataFrame(all_data)
    if save:
        outfile = join(indir, 'evaluations.all.hdf')
        save_df(dfr, outfile, key='evaluation_all')
    return dfr


def aggeval_html(indir, save=True):
    template_chunks = []
    all_data = []
    for (html_pre, data_list, classes, weights, columns, html_post) in \
            _evalhtmlscanner(indir):
        all_data.extend(data_list)
        if not template_chunks:
            template_chunks = [html_pre, '\n', html_post]

    template_chunks[1] = json.dumps(all_data)
    content = ''.join(template_chunks)
    re_title = re.compile(r'<title>(.*?)</title>',
                          re.IGNORECASE)  # @UndefinedVariable
    oldtitle = re_title.search(content).group(1).strip()
    newtitle = "Summary CV evaluations"
    content = content.replace(oldtitle, newtitle)
    if save:
        outfile = join(indir, 'evaluations.all.html')
        with open(outfile, 'w') as _opn:
            _opn.write(content)
    return content


def _evalhtmlscanner(indir):
    dotall, icase = re.DOTALL, re.IGNORECASE  # @UndefinedVariable
    re_data = re.compile(r'evaluations: +(\[\{.*?\}\]),\n', dotall)
    html_pre, html_post = '', ''

    jsvars = {
        're': {
            'classes': re.compile(r'classes: +(\[.*?\]),\n', dotall),
            'weights': re.compile(r'weights: +(\[.*?\]),\n', dotall),
            'columns': re.compile(r'columns: +(\[.*?\]),\n', dotall)
        },
        'val': {
            'classes': [],
            'weights': [],
            'columns': []
        }
    }

    for fle in listdir(indir):
        bfle, efle = splitext(fle)
        if efle.lower() == '.html':
            if '?' not in bfle:
                continue
            prefix = "clf=%s" % bfle[:bfle.index('?')]
            params = ParamsEncDec.todict(bfle)
            if 'features' not in params or len(params) > 1:
                continue
            prefix += " features=%s" % ",".join(params['features'])
            all_data = []
            with open(join(indir, fle), 'r') as _opn:
                content = _opn.read()
                match_data = re_data.search(content)
                if not match_data:
                    raise ValueError('No data match found')
                try:
                    for dic in json.loads(match_data.group(1)):
                        dic['key'] = prefix + " " + dic['key']
                        all_data.append(dic)
                except:  # @IgnorePep8
                    raise ValueError('data unparsable as JSON')
                start, end = match_data.start(1), match_data.end(1)
                html_pre, html_post = content[:start], content[end:]
                for name, matcher in jsvars['re'].items():
                    match_ = matcher.search(content)
                    if not match_:
                        raise ValueError('No match found for variable "%s"' %
                                         name)
                    try:
                        _pyval = json.loads(match_.group(1))
                    except:  # @IgnorePep8
                        raise ValueError('variable "%s" unparsable as JSON' %
                                         name)
                    if not jsvars['val'][name]:
                        jsvars['val'][name] = _pyval
                    elif jsvars['val'][name] != _pyval:
                        raise ValueError('Variable "%s" not equal in all '
                                         'html files' % name)
                yield (html_pre, all_data, *jsvars['val'].values(),
                       html_post)


class Evaluator:
    '''Class for evaluating pre-fitted and saved model(s) (ususally obtained
    via `CVEvaluator`) against a
    dataset of instances, saving to file the predictions and an html report
    of the classifiers performances
    '''
    def __init__(self, classifier_paths, normalizer_df=None):
        if len(set(basename(clfpath) for clfpath in classifier_paths)) != \
                len(classifier_paths):
            raise ValueError('You need to pass a list of unique file names')

        self.clf_features = {}
        for clfpath in classifier_paths:
            fpath, ext = splitext(clfpath)
            if ext != '.model':
                raise ValueError('Classifiers file names must have extension'
                                 ' .model')
            _params = ParamsEncDec.todict(fpath)
            if 'features' not in _params:
                raise ValueError("'features=' not found in file name '%s'" %
                                 basename(fpath))
            self.clf_features[basename(clfpath)] = _params['features']

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
            for features in self.clf_features.values():
                for feat in features:
                    bounds[feat] = (None, None)
            normalizer_df_columns = set(normalizer_df.columns)
            ndf = normalizer_df[~is_outlier(normalizer_df)]
            for feat in list(bounds.keys()):
                if feat not in normalizer_df_columns:
                    raise ValueError('"%s" not in normalizer dataframe' % feat)
                bounds[feat] = np.nanmin(ndf[feat]), np.nanmax(ndf[feat])
        self.bounds = bounds

    def run(self, test_df, destdir):
        test_dataset_name = dataset_info(test_df).__name__

        predictions_destdir = abspath(join(destdir,
                                           CVEvaluator.PREDICTIONSDIRNAME))
        if not isdir(predictions_destdir):
            makedirs(predictions_destdir)
        if not isdir(predictions_destdir):
            raise ValueError('Unable to create dirctory "%s"' %
                             predictions_destdir)

        print('Evaluating %d classifiers on test dataset "%s"'
              % (len(self.clfs), test_dataset_name))

        pool = Pool(processes=int(cpu_count()))

        with click.progressbar(
            length=len(self.clfs),
            fill_char='o', empty_char='.'
        ) as pbar:

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
                        pool.imap_unordered(_imap_predict,
                                            self.iterator(test_df)):
                    pbar.update(1)
                    pred_basename = splitext(clfname)[0]
                    pred_basename += "&testset=%s" % test_dataset_name + '.hdf'
                    save_df(predicted_df,
                            join(predictions_destdir, pred_basename),
                            key='predictions')
                    key = splitext(clfname)[0].replace('?', ' ').\
                        replace('&', ' ')
                    cmatrix_dfs['clf=%s' % key] = cmatrix_df(predicted_df)

                pool.close()
                pool.join()

            except Exception as exc:  # pylint: disable=broad-except
                kill_pool(exc)

            outfilepath = join(destdir, 'evalreport.html')
            title = ('Evalation results comparing '
                     '%d classifiers on test dataset "%s"') % \
                    (len(self.clfs), test_dataset_name)
            return create_evel_report(cmatrix_dfs, outfilepath, title=title)

    def iterator(self, dataframe):
        '''Yields tuples of (name, clf, dataframe, features) for all
        classifiers of this class
        '''
        for name, clf in self.clfs.items():
            features = self.clf_features[name]
            dataframe_ = drop_duplicates(dataframe, features, 0, verbose=False)
            dataframe_ = keep_cols(dataframe_, features).copy()
            # normalize:
            if self.bounds is not None:
                for feat in features:
                    min_, max_ = self.bounds[feat]
                    dataframe_.loc[:, feat] = \
                        (dataframe_[feat] - min_) / (max_ - min_)

            yield name, clf, dataframe_, features


def _imap_predict(arg):
    '''Predicts teh given classifier and returns the classifier name and
    the prediction dataframe. Called from within imap in Evaluator
    '''
    name, clf, dfr, feats = arg
    return name, predict(clf, dfr, *feats)


class ParamsEncDec:
    '''This class converts to and from dicts of parameters and
    query strings, in a fashion very similar to GET and POST requests data

    THIS CLASS IS ONLY INTENDED TO GENERATE UNIQUE STRINGS FROM DICTS OF
    PARAMETERS: WHEN GETTING THE DICT BACK FROM THOSE STRINGS, REMEMBER THAT
    VALUES ARE ALL STRINGS, THUS IN THE CONVERSION DICT -> STR ->DICT,
    DICT VALUE TYPES ARE NOT PRESERVED

    ParamsEncDec.tostr(dict) -> produces query string: "?p1=v1&p2=v2&.."
    ParamsEncDec.todict(str) -> produces dict: {'a': '1,2,3', 'b', 'r', ...}

    '''
    # define two dict for percent encode. This is done also by urllib.quote
    # does the same but we skip it as we want to encode only few characters:
    _chr2encode = {k: v for k, v in zip('/&?=,',
                                        ['%2F', '%26', '%3F', '%3D', '%2C'])}

    @staticmethod
    def tostr(*features, **params):
        '''Encodes params dict to string in a URL query fashion:
        ?param1=value&param2=value2...

        :param features: a list of features
        :param params: dict of string params mapped to values (pass OrederdDict
            if you want to preserve insertion order and Python < 3.7)
        '''

        def quote(string):
            '''percent encodes some characters only'''
            return ''.join(ParamsEncDec._chr2encode.get(c, c) for c in string)

        chunks = []
        prefix = '?'
        if features:
            chunks.append("%sfeatures=%s" %
                          (prefix, ",".join(quote(str(_)) for _ in features)))
            prefix = '&'
        for key, val in params.items():
            chunks.append('%s%s=%s' %
                          (prefix, quote(str(key)), quote(str(val))))
            prefix = '&'

        return ''.join(chunks)

    @staticmethod
    def todict(filepath_noext):
        '''Decodes string encoded parameters to dict

        :param string: string including a query string denoting parameters
            used (encoded with `tostr` above). E.g., it can be
            a full file path:
            /user/me/file?a=8,9,10&b=gamma
            a file name:
            file?a=8,9,10&b=gamma
            or simply its query string portion:
            ?a=8,9,10&b=gamma
        '''

        def unquote(string):
            '''decodes percent encodeed characters, but only few cases only'''
            for char, percentenc_str in ParamsEncDec._chr2encode.items():
                string = string.replace(percentenc_str, char)
            return string

        ret = odict()  # ordered dict
        pth = basename(filepath_noext)
        if '?' not in pth:
            raise ValueError('"?" not in "%s"' % pth)
        pth = pth[pth.find('?') + 1:]
        splits = pth.split('&')
        for chunk in splits:
            param, value = chunk.split('=')
            if param == 'features':
                value = tuple(unquote(_) for _ in value.split(','))
            else:
                value = unquote(value)
            ret[unquote(param)] = value
        return ret
