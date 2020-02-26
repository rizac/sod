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

from urllib.parse import quote as q, unquote as uq
import click
import numpy as np
import pandas as pd
from pandas.core.indexes.range import RangeIndex
from joblib import dump, load
from sklearn.metrics import (confusion_matrix as scikit_confusion_matrix,
                             log_loss as scikit_log_loss,
                             roc_curve as scikit_roc_curve,
                             precision_recall_curve as scikit_pr_curve,
                             auc as scikit_auc,
                             average_precision_score as scikit_aps)

from sod.core import pdconcat, odict, CLASS_SELECTORS, CLASSNAMES
from sod.core.dataset import is_outlier, OUTLIER_COL
import re
from sklearn.ensemble.iforest import IsolationForest
from sklearn.svm.classes import OneClassSVM
import importlib
from sklearn.metrics.classification import precision_recall_fscore_support
import types


PREDICT_COL = 'predicted_anomaly_score'


def classifier(clf_class, dataframe, clf_params, destpath=None, overwrite=False):
    '''Returns a OneClassSVM classifier fitted with the data of
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
        empty list/tuple: use all columns

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


def log_loss(predicted_df, eps=1e-15, return_mean=True):
    '''Computes the log loss of `predicted_df`

    :param predicted_df: A dataframe with predictions, the output of
        `predict`
    :param return_mean: bool, optional (default=True)
        If true, return the mean loss per sample.
        Otherwise, return the sum of the per-sample losses.

    :return: a NUMBER representing the mean (normalize=True) or sum
        (normalize=False) of all scores in predicted_df
    '''
    return scikit_log_loss(predicted_df[OUTLIER_COL],
                           predicted_df[PREDICT_COL],
                           eps=eps, normalize=return_mean,
                           labels=[False, True])


def confusion_matrix(predicted_df, threshold=0.5, compute_eval_metrics=True):
    y_pred = predicted_df[PREDICT_COL] > threshold
    return confusion_matrix_prfs(predicted_df.outlier,
                                 y_pred,
                                 labels=[False, True],
                                 compute_eval_metrics=compute_eval_metrics)


def confusion_matrix_prfs(y_true, y_pred, labels=None,
                          compute_eval_metrics=False):
    cm = scikit_confusion_matrix(y_true,
                                 y_pred,
                                 labels=labels)
    dfr = pd.DataFrame(data=cm, index=CLASSNAMES, columns=CLASSNAMES)
    if compute_eval_metrics:
        prfs = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                               average=None)
        assert (prfs[-1] == dfr.sum(axis=1)).all()
        P, R, F, S = 'Precision', 'Recall', 'F1Score', 'Support'
        dfr[S] = prfs[-1]
        dfr[R] = prfs[1]
        dfr[P] = prfs[0]
        dfr[F] = prfs[2]

        # attach bound methods: https://stackoverflow.com/a/2982
        # dfr.precision(), dfr.recall() etcetera
        dfr.p = dfr.precisions = dfr.precision = \
            types.MethodType(lambda s: s.loc[:, P], dfr)
        dfr.r = dfr.recalls = dfr.recall = \
            types.MethodType(lambda s: s.loc[:, R], dfr)
        dfr.f = dfr.f1scores = dfr.f1score = \
            types.MethodType(lambda s: s.loc[:, F], dfr)
        dfr.s = dfr.supports = dfr.support = \
            types.MethodType(lambda s: s.loc[:, S], dfr)
        dfr.num_instances = types.MethodType(lambda s: s.loc[:, S], dfr)

    return dfr


def best_threshold(y_true, y_score, method='roc'):
    if method == 'roc':
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve

        fpr, tpr, thresholds = scikit_roc_curve(y_true, y_score, pos_label=1)
        # Convert to TNR (avoid dividing by 2 as useless):
        x_tnr = 1 - fpr
        # get the best threshold where we have the best mean of TPR and TNR:
        scores = x_tnr + tpr
        # Get tbest threshold ignoring 1st score. From the docs (see linke
        # above): thresholds[0] represents no instances being predicted and
        # is arbitrarily set to max(y_score) + 1.
        best_th_index = np.argmax(scores[1:])
        return thresholds[1:][best_th_index]

    if method == 'pr':
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve

        prc, rec, thresholds = scikit_pr_curve(y_true, y_score, pos_label=1)

        # get the best threshold where we have the best F1 score
        # (avoid multiplying by 2 as useless):
        scores = (prc * rec) / (prc + rec)
        # Get best score ignoring lat score. From the docs (see link above):
        # the last precision and recall values are 1. and 0. respectively and
        # do not have a corresponding threshold. This ensures that the graph
        # starts on the y axis.
        best_th_index = np.argmax(scores[:-1])
        return thresholds[:-1][best_th_index]

    raise ValueError('`method` argument in `best_threshold` must be '
                     'either "roc" (ROC curve) or '
                     '"pr" (Precision-Recall Curve)')
        

def auc(y_true, y_score):
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc
    fpr, tpr, thresholds = scikit_roc_curve(y_true, y_score, pos_label=1)
    return scikit_auc(fpr, tpr)


def aps(y_true, y_score):
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
    return scikit_aps(y_true, y_score, pos_label=1)


# def _get_eval_report_html_template():
#     with open(join(dirname(__file__), 'eval_report_template.html'), 'r') as _:
#         return _.read()
# 
# 
# def create_evel_report(cmatrix_dfs, outfilepath=None, title='%(title)s'):
#     '''Creates and optionally saves the given confusion matrices dataframe
#     to html
# 
#     :param cmatrix_df: a dict of string keys
#         mapped to dataframe as returned from the function `cmatrix_df`
#     :param outfilepath: the output file path. The extension will be
#         appended as 'html', if an extension is not set. If None, nothing is
#         saved
#     :param title: the HTML title page
# 
#     :return: the HTML formatted string containing the evaluation report
#     '''
#     # score columns: list of 'asc', 'desc' or None relative to each element of
#     # CMATRIX_COLUMNS
#     score_columns = {
#         CMATRIX_COLUMNS[-2]: 'desc',  # '% rec.'
#         CMATRIX_COLUMNS[-1]: 'asc'  # 'Mean %s' % LOGLOSS_COL
#     }
#     evl = [
#         {'key': params, 'data': sumdf.values.tolist()}
#         for (params, sumdf) in cmatrix_dfs.items()
#     ]
#     classnames = CLASSNAMES
#     content = _get_eval_report_html_template() % {
#         'title': title,
#         'evaluations': json.dumps(evl),
#         'columns': json.dumps(CMATRIX_COLUMNS),
#         'scoreColumns': json.dumps(score_columns),
#         'currentScoreColumn': json.dumps(CMATRIX_COLUMNS[-1]),
#         'weights': json.dumps([100 for _ in classnames]),
#         'classes': json.dumps(classnames)
#     }
#     if outfilepath is not None:
#         if splitext(outfilepath)[1].lower() not in ('.htm', '.html'):
#             outfilepath += ' .html'
#         with open(outfilepath, 'w') as opn_:
#             opn_.write(content)
# 
#     return content


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


class TrainingParam:
    '''TrainingParam class to be used in Evaluator
    Udage:
    
    t = TrainingParam(args)
    for args in t(destdir):
        _create_save_classifier(args)
    '''
    def __init__(self, clf_classname, clf_param_dict,
                 input_filenpath, input_features_list, input_drop_na):
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

        if not isfile(input_filenpath):
            raise FileNotFoundError(input_filenpath)
        self.input_filepath = input_filenpath
        self.features = input_features_list
        self.drop_na = input_drop_na

    def iterargs(self, destdir):
        ret = []
        for features in self.features:
            for params in self.parameters:
                destpath = join(destdir,
                                model_filename(self.clf_class,
                                               basename(self.input_filepath),
                                               *features, **params))
                ret.append((
                    self.clf_class,
                    self.input_filepath,
                    features,
                    params,
                    destpath,
                    self.drop_na
                ))
        return ret

#     def uniquefilepath(self, destdir, *features, **params):
#         '''Returns an unique file path from the given features and params'''
#         # build an orderd dict
#         paramz = odict()
#         # add them to paramz:
#         for k in sorted(k for k in params):
#             paramz[k] = params[k]
#         # build a base file path with the current classifier class name:
#         basepath = join(destdir, self.clf_class.__name__)
#         # add the URLquery-like string with ParamsEncDec.tostr:
#         return basepath + ParamsEncDec.tostr(*features, **paramz)


def _create_save_classifier(args):
    (clf_class, input_filepath, features, params, destpath, drop_na) = args
    if not isdir(dirname(destpath)):
        raise ValueError('Can not store model, parent directory does not exist: '
                         '"%s"' % destpath)
    elif isfile(destpath):
        return destpath, False
    dataframe = pd.read_hdf(input_filepath, columns=features)
    if drop_na:
        dataframe = dataframe.dropna(axis=0, subset=features, how='any')
    clf = classifier(clf_class, dataframe, params)
    dump(clf, destpath)
    return destpath, True


class TestParam:
    
    def __init__(self, input_filepath, categorical_columns, columns2save, drop_na):
        if not isfile(input_filepath):
            raise FileNotFoundError(input_filepath)
        self.input_filepath = input_filepath
        self.categorical_columns = categorical_columns
        self.columns2save = columns2save
        self.drop_na = drop_na
        
    def iterargs(self, classifiers_paths):
        ret = []
        for clfpath in classifiers_paths:
            outdir = splitext(clfpath)[0]
            outfile = join(outdir, basename(self.input_filepath))
            if isfile(outfile):
                continue
            feats = model_params(clfpath)['feats']
            ret.append([feats, clfpath, self.input_filepath,
                        self.categorical_columns, self.columns2save, self.drop_na,
                        outfile])
        return ret


def _predict_and_save(args):
    (features, clfpath, input_filepath, categorical_columns, columns2save, drop_na,
     outfile) = args
    allcols = list(columns2save) + list(_ for _ in features if _ not in columns2save)
    dataframe = pd.read_hdf(input_filepath, columns=allcols)
    for c in categorical_columns or []:
        if c in allcols:
            dataframe[c] = dataframe[c].astype('category')
    if drop_na:
        dataframe = dataframe.dropna(axis=0, subset=features, how='any')
    pred_df = predict(load(clfpath), dataframe, features, columns2save)
    if not isdir(dirname(outfile)):
        makedirs(dirname(outfile))
    save_df(pred_df, outfile)
    return outfile


# make safe all characters except chars <= ' ' (32) and '/&?=,'
_safe = ''.join(set(chr(_) for _ in range(33, 127)) - set('/&?=,%'))

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
    iterargs = training_param.iterargs(destdir)
    with click.progressbar(length=len(iterargs),
                           fill_char='o', empty_char='.') as pbar:
        try:
            newly_created_models = 0
            for clfpath, newly_created in \
                    pool.imap_unordered(_create_save_classifier, iterargs):
                newly_created_models += newly_created
                classifier_paths.append(clfpath)
                pbar.update(1)
            # absolutely call these methods, although in impa and imap unordered
            # do make sense?
            pool.close()
            pool.join()
        except Exception as exc:
            _kill_pool(pool, str(exc))
            raise exc
    print("%d of %d models created (already existing were not overwritten)" %
          (newly_created_models, len(classifier_paths)))

    print('Step 2 of 2: Testing (creating prediction data frames)')
    pool = Pool(processes=int(cpu_count()))
    iterargs = testing_param.iterargs(classifier_paths)
    pred_filepaths = []
    with click.progressbar(length=len(iterargs), fill_char='o', empty_char='.') as pbar:
        try:
            for pred_filepath in pool.imap_unordered(_predict_and_save, iterargs):
                if pred_filepath:
                    pred_filepaths.append(pred_filepath)
                pbar.update(1)
            # absolutely call these methods, although in impa and imap unordered
            # do make sense?
            pool.close()
            pool.join()
        except Exception as exc:
            _kill_pool(pool, str(exc))
            raise exc     
    print("%d of %d predictions created (already existing were not overwritten)" %
          (len(pred_filepaths), len(classifier_paths)))

    print()
    print('DONE')


def _kill_pool(pool, err_msg):
    print('ERROR:')
    print(err_msg)
    try:
        pool.terminate()
    except ValueError:  # ignore ValueError('pool not running')
        pass


# AGGEVAL_BASENAME = 'evaluation.all'
