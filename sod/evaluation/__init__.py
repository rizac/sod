'''
    Evaluation commmon utilities
'''
import sys
import json
from contextlib import contextmanager
from multiprocessing import Pool, cpu_count
from io import StringIO
from os import makedirs
from os.path import abspath, join, dirname, isfile, isdir, basename
from itertools import repeat, product
import warnings
from collections import defaultdict
from datetime import timedelta

from joblib import dump, load
import numpy as np
import pandas as pd
import time
from sklearn.metrics.classification import confusion_matrix
import click
from pandas.core.indexes.range import RangeIndex


DATASET_COLUMNS = [
    "station_id",
    "event_time",
    "amplitude_ratio",
    "snr",
    "magnitude",
    "distance_km",
    "pga_observed",
    "pga_predicted",
    "pgv_observed",
    "pgv_predicted",
    "noise_psd@1sec",
    "noise_psd@2sec",
    "noise_psd@3sec",
    "noise_psd@5sec",
    "noise_psd@9sec",
    "amp@0.5hz",
    "amp@1hz",
    "amp@2hz",
    "amp@5hz",
    "amp@10hz",
    "amp@20hz",
    "outlier",
    "modified",
    "Segment.db.id",
    "delta_pga",
    "delta_pgv"
]

DATASET_MODIFICATION_TYPES: [
    ""
    "STAGEGAIN:X2.0",
    "STAGEGAIN:X10.0",
    "STAGEGAIN:X100.0",
    "STAGEGAIN:X0.01",
    "STAGEGAIN:X0.1",
    "STAGEGAIN:X0.5",
    "CHARESP:LHZ",
    "CHARESP:LHN",
    "CHARESP:LHE",
    "CHARESP:HHZ",
    "CHARESP:HHN",
    "CHARESP:HHE",
    "CHARESP:BHZ",
    "CHARESP:BHN",
    "CHARESP:BHE",
    "CHARESP:HNZ",
    "CHARESP:HNE",
    "CHARESP:HNN",
    "INVFILE:FR.PYLO.2010-01-17T10:00:00.xml",
    "CHARESP:VHE",
    "CHARESP:VHN",
    "CHARESP:VHZ",
    "CHARESP:EHE",
    "CHARESP:EHN",
    "CHARESP:EHZ",
    "CHARESP:HLE",
    "CHARESP:HLN",
    "CHARESP:HLZ",
    "INVFILE:CH.GRIMS.2015-10-30T10:50:00.xml",
    "CHARESP:HGE",
    "CHARESP:HGN",
    "CHARESP:HGZ",
    "INVFILE:CH.GRIMS.2011-11-09T00:00:00.xml",
    "CHARESP:SHZ",
    "CHARESP:SNE",
    "CHARESP:SNN",
    "CHARESP:SNZ",
    "CHARESP:SHE",
    "CHARESP:SHN",
    "INVFILE:SK.MODS.2004-03-17T00:00:00.xml",
    "INVFILE:SK.ZST.2004-03-17T00:00:00.xml",
    "CHARESP:BNZ",
    "CHARESP:LNZ",
    "CHARESP:LNN",
    "CHARESP:BNN",
    "CHARESP:LNE",
    "CHARESP:BNE"
]


DATASET_FILENAME = abspath(join(dirname(__file__), '..',
                                'dataset', 'dataset.hdf'))


@contextmanager
def capture_stderr(verbose=False):
    '''Context manager to be used in a with statement in order to capture
    std.error messages (e.g., python warnings):
    ```
    with capture_stderr():
        ... code here
    ```
    :param verbose: boolean (default False). If True, prints the captured
        messages (if present)
    '''
    # Code to acquire resource, e.g.:
    # capture warnings which are redirected to stderr:
    syserr = sys.stderr
    captured_err = StringIO()
    sys.stderr = captured_err
    try:
        yield
        if verbose:
            errs = captured_err.getvalue()
            if errs:
                print('')
                print('During the operation, '
                      'the following warning(s) were issued:')
                print(errs)
        captured_err.close()
    finally:
        # restore standard error:
        sys.stderr = syserr


def open_dataset(filename=None, verbose=True):
    if filename is None:
        filename = DATASET_FILENAME
    if verbose:
        print('Opening %s' % abspath(filename))

    # capture warnings which are redirected to stderr:
    with capture_stderr(verbose):
        dfr = pd.read_hdf(filename)

        # setting up columns:
        dfr['pga'] = np.log10(dfr['pga_observed'].abs())
        dfr['pgv'] = np.log10(dfr['pgv_observed'].abs())
        dfr['delta_pga'] = np.log10(dfr['pga_observed'].abs()) - \
            np.log10(dfr['pga_predicted'].abs())
        dfr['delta_pgv'] = np.log10(dfr['pgv_observed'].abs()) - \
            np.log10(dfr['pgv_predicted'].abs())
        del dfr['pga_observed']
        del dfr['pga_predicted']
        del dfr['pgv_observed']
        del dfr['pgv_predicted']
        for col in dfr.columns:
            if col.startswith('amp@'):
                # go to db. We should multuply log * 20 (amp spec) or * 10 (pow spec)
                # but it's unnecessary as we will normalize few lines below
                dfr[col] = np.log10(dfr[col])
        # save space:
        dfr['modified'] = dfr['modified'].astype('category')
        # numpy int64 for just zeros and ones is waste of space: use bools (int8)
        # let's be paranoid first (check later, see below)
        _zum = dfr['outlier'].sum()
        # convert:
        dfr['outlier'] = dfr['outlier'].astype(bool)
        # check:
        if dfr['outlier'].sum() != _zum:
            raise ValueError('The column "outlier" is supposed to be populated with '
                             '0 or 1, but conversion to boolean failed. Check the column')

        if verbose:
            print('')
            print(dfinfo(dfr))

        sum_df = {}
        if verbose:
            print('')
            print('Normalizing numeric columns (floats only)')
        columns = ['min', 'median', 'max', 'NAs', 'segs1-99',
                   'stas1-99']
        oks_ = ~is_outlier(dfr)
        for col in floatingcols(dfr):
            _dfr = dfr.loc[oks_, :]
            q01 = np.nanquantile(_dfr[col], 0.01)
            q99 = np.nanquantile(_dfr[col], 0.99)
            df1, df99 = _dfr[(_dfr[col] <= q01)], _dfr[(_dfr[col] >= q99)]
            segs1 = len(pd.unique(df1['Segment.db.id']))
            segs99 = len(pd.unique(df99['Segment.db.id']))
            stas1 = len(pd.unique(df1['station_id']))
            stas99 = len(pd.unique(df99['station_id']))

            # for calculating min and max, we need to drop also infinity, tgus
            # np.nanmin and np.nanmax do not work. Hence:
            finite_values = _dfr[col][np.isfinite(_dfr[col])]
            min_, max_ = np.min(finite_values), np.max(finite_values)
            dfr[col] = (dfr[col] - min_) / (max_ - min_)
            if verbose:
                sum_df[col] = {
                    columns[0]: dfr[col].min(),
                    columns[1]:  dfr[col].quantile(0.5),
                    columns[2]: dfr[col].max(),
                    columns[3]: (~np.isfinite(dfr[col])).sum(),
                    columns[4]: segs1 + segs99,
                    columns[5]: stas1 + stas99,
                }
        if verbose:
            print(df2str(pd.DataFrame(data=list(sum_df.values()),
                                      columns=columns,
                                      index=list(sum_df.keys()))))
            print("-------")
            print("Min and max might be outside [0, 1]: the normalization ")
            print("bounds are calculated on good segments (non outlier) only")
            print("%s: values which are NaN or Infinity" % columns[3])
            print("%s: good segments (not outliers) "
                  "outside 1 percentile" % columns[4])
            print("%s: good stations (with segments not outliers) "
                  "outside 1 percentile" % columns[5])

        # for safety:
        dfr.reset_index(drop=True, inplace=True)
        return dfr


def groupby_stations(dataframe, verbose=True):
    '''Groups `dataframe` by stations and returns the resulting dataframe
    Numeric columns are merged taking the median of all rows
    '''
    if verbose:
        print('Grouping dataset per station')
        print('(For floating columns, the median of all segments stations '
              'will be set)')
        print('')
    with capture_stderr(verbose):
        newdf = []
        fl_cols = list(floatingcols(dataframe))
        for (staid, modified, outlier), _df in \
                dataframe.groupby(['station_id', 'modified', 'outlier']):
            _dfmedian = _df[fl_cols].median(axis=0, numeric_only=True, skipna=True)
            _dfmedian['num_segments'] = len(_df)
            _dfmedian['outlier'] = outlier
            _dfmedian['modified'] = modified
            _dfmedian['station_id'] = staid
            newdf.append(pd.DataFrame([_dfmedian]))
            # print(pd.DataFrame([_dfmedian]))

        ret = pdconcat(newdf, ignore_index=True)
        ret['num_segments'] = ret['num_segments'].astype(int)
        # convert dtypes because they might not match:
        shared_c = (set(dataframe.columns) & set(ret.columns)) - set(fl_cols)
        for col in shared_c:
            ret[col] = ret[col].astype(dataframe[col].dtype)
        if verbose:
            bins = [1, 10, 100, 1000, 10000]
            if ret['num_segments'].max() > bins[-1]:
                bins.append(ret['num_segments'].max() + 1)
            groups = ret.groupby(pd.cut(ret['num_segments'], bins, precision=0,
                                        right=False))
            print(pd.DataFrame(groups.size(), columns=['num_stations']).
                  to_string())
            assert groups.size().sum() == len(ret)
            print('')
            print('Summary of the new dataset (instances = stations)')
            print(dfinfo(ret))
        return ret


def floatingcols(dataframe):
    '''Iterable yielding all floating point columns of dataframe'''
    for col in dataframe.columns:
        try:
            if np.issubdtype(dataframe[col].dtype, np.floating):
                yield col
        except TypeError:
            # categorical data falls here
            continue


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
        print(dfinfo(dataframe))
    return dataframe


def drop_na(dataframe, columns, verbose=True):
    '''
    Remove rows with non finite values.under the specified columns
        (a value is finite when it is neither NaN nor +-Infinity)

    :return: a VIEW of dataframe with only rows with finite values. If you want to
        modify the returned DataFrame, call copy() on it
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
            print(dfinfo(dataframe))

    return dataframe


def keep_cols(dataframe, columns):
    '''
    Drops all columns of dataframe not in `columns` (preserving in any case the
    columns 'outlier', 'modified' and 'Segment.db.id')

    :return: a VIEW of the original dataframe (does NOT return a copy) keeping the
        specified columns only
    '''
    cols = list(columns) + ['outlier', 'modified']
    if 'Segment.db.id' in dataframe.columns:
        cols += ['Segment.db.id']
    return dataframe[list(set(cols))]


def classifier(clf_class, dataframe, **clf_params):
    '''Returns a OneClassSVM classifier fitted with the data of
    `dataframe`.

    :param dataframe: pandas DataFrame
    :param columns: list of string denoting the columns of `dataframe` that represents
        the feature space to fit the classifier with
    :param clf_params: parameters to be passed to `clf_class`
    '''
    clf = clf_class(**clf_params)
    clf.fit(dataframe.values)
    return clf


def predict(clf, dataframe, *columns):
    '''
    :return: a DataFrame with columns 'correctly_predicted' (boolean) plus
        the three columns copied from `dataframe` useful to uniquely identify the
        segment: 'outlier' (boolean), 'modified' (categorical)
        and 'Segment.db.id' (int).
    '''
    predicted = _predict(clf, dataframe if not columns else dataframe[list(columns)])
    label = np.ones(len(predicted), dtype=predicted.dtype)
    label[is_outlier(dataframe)] = - 1
    correctly_predicted = label == predicted
    data = {
        'correctly_predicted': correctly_predicted,
        'outlier': dataframe['outlier'],
        'modified': dataframe['modified']
    }
    if 'Segment.db.id' in dataframe.columns:
        data['Segment.db.id'] = dataframe['Segment.db.id']
    return pd.DataFrame(data, index=dataframe.index)


def _predict(clf, dataframe):
    '''Returns a numpy array of len(dataframe) integers in [-1, 1],
    where:
    -1: item is classified as OUTLIER
     1: item is classified as OK (no outlier)
    Each number at index I is the prediction (classification) of
    the I-th element of `dataframe`.

    :param clf: the given (trained) classifier
    :param dataframe: pandas DataFrame
    :param columns: list of string denoting the columns of `dataframe` that
        represents the feature space to fit the classifier with
    '''
    # this method is very trivial it is used mainly for test purposes (mock)
    return clf.predict(dataframe.values)


def cmatrix(dataframe, sample_weights=None):
    '''
        :param dataframe: the output of predict(dataframe, *columns)
        :param sample_weights: a numpy array the same length of dataframe
            with the sameple weights. Default: None (no weights)

        :return: a pandas 2x2 DataFrame with labels 'ok', 'outlier'
    '''
    labelnames = ['ok', 'outlier']
    labels = np.ones(len(dataframe))
    labels[is_outlier(dataframe)] = -1
    predicted = -labels
    predicted[dataframe['correctly_predicted']] = \
        labels[dataframe['correctly_predicted']]
    confm = pd.DataFrame(confusion_matrix(labels,
                                          predicted,
                                          # labels here just assures that
                                          # 1 (ok) is first and
                                          # -1 (outlier) is second:
                                          labels=[1, -1],
                                          sample_weight=sample_weights),
                         index=labelnames, columns=labelnames)
    confm.columns.name = 'Classified as:'
    confm.index.name = 'Label:'
    return confm


def is_outlier(dataframe):
    '''pandas series of boolean telling where dataframe rows are outliers'''
    return dataframe['outlier']  # simply return the column


def is_out_wrong_inv(dataframe):
    '''pandas series of boolean telling where dataframe rows are outliers
    due to wrong inventory
    '''
    return dataframe['modified'].str.contains('INVFILE:')


def is_out_swap_acc_vel(dataframe):
    '''pandas series of boolean telling where dataframe rows are outliers
    generated by swapping the accelerometer and velocimeter
    response in the inventory (when the segment's inventory has both
    accelerometers and velocimeters)
    '''
    return dataframe['modified'].str.contains('CHARESP:')


def is_out_gain_x10(dataframe):
    '''pandas series of boolean telling where dataframe rows are outliers
    generated by multiplying the trace by a factor of 100 (or 0.01)
    '''
    return dataframe['modified'].str.contains('STAGEGAIN:X10.0') | \
        dataframe['modified'].str.contains('STAGEGAIN:X0.1')


def is_out_gain_x100(dataframe):
    '''pandas series of boolean telling where dataframe rows are outliers
    generated by multiplying the trace by a factor of 10 (or 0.1)
    '''
    return dataframe['modified'].str.contains('STAGEGAIN:X100.0') | \
        dataframe['modified'].str.contains('STAGEGAIN:X0.01')


def is_out_gain_x2(dataframe):
    '''pandas series of boolean telling where dataframe rows are outliers
    generated by multiplying the trace by a factor of 2 (or 0.5)
    '''
    return dataframe['modified'].str.contains('STAGEGAIN:X2.0') | \
        dataframe['modified'].str.contains('STAGEGAIN:X0.5')


def pdconcat(dataframes, **kwargs):
    '''forwards to pandas concat with standard arguments'''
    return pd.concat(dataframes, sort=False, axis=0, copy=True, **kwargs)


def split(size, n_folds):
    '''Iterable yielding tuples of `(start, end)` indices
    (`start` < `end`) resulting from splitting `size` into `n_folds`.
    Both start and end are >= than 0 and <= `size`
    '''
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
        yield dataframe.loc[~dataframe.index.isin(samples), :].copy(), \
            dataframe.loc[samples, :].copy()


def df2str(dataframe):
    ''':return: the string representation of `dataframe`, with numeric values
    formatted with comma as decimal separator
    '''
    return dfformat(dataframe).to_string()


def dfformat(dataframe):
    strformat = {c: "{:,d}" if str(dataframe[c].dtype).startswith('int')
                 else '{:,.2f}' for c in dataframe.columns}
    return pd.DataFrame({c: dataframe[c].map(strformat[c].format)
                         for c in dataframe.columns},
                        index=dataframe.index)


def dfinfo(dataframe, perclass=True):
    columns = ['instances']
    if not perclass:
        oks = ~is_outlier(dataframe)
        oks_count = oks.sum()
        data = [oks_count, len(dataframe)-oks_count, len(dataframe)]
        index = ['oks', 'outliers', 'total']
    else:
        data = [_(dataframe).sum() for _ in CLASSES.values()] + [len(dataframe)]
        index = list(CLASSES.keys()) + ['total']

    return df2str(pd.DataFrame(data, columns=columns, index=index))


_CLASSNAMES = [
    'ok',
    'outl. (wrong inv. file)',
    'outl. (cha. resp. acc <-> vel)',
    'outl. (gain X100 or X0.01)',
    'outl. (gain X10 or X0.1)',
    'outl. (gain X2 or X0.5)'
]

CLASSES = {
    _CLASSNAMES[0]: lambda dfr: ~is_outlier(dfr),
    _CLASSNAMES[1]: is_out_wrong_inv,
    _CLASSNAMES[2]: is_out_swap_acc_vel,
    _CLASSNAMES[3]: is_out_gain_x100,
    _CLASSNAMES[4]: is_out_gain_x10,
    _CLASSNAMES[5]: is_out_gain_x2
}

WEIGHTS = {
    _CLASSNAMES[0]: 100,
    _CLASSNAMES[1]: 100,
    _CLASSNAMES[2]: 10,
    _CLASSNAMES[3]: 50,
    _CLASSNAMES[4]: 5,
    _CLASSNAMES[5]: 1
}


class EvalResult:

    __slots__ = ['clf', 'predictions', 'params', 'features']

    def __init__(self, clf, params, features):
        self.clf = clf
        self.params = params
        self.predictions = None
        self.features = features

    def predict(self, dataframe):
        self.predictions = \
            None if dataframe.empty else predict(self.clf, dataframe, *self.features)


def fit_and_predict(clf_class, train_df, columns, params, test_df=None,
                    filepath=None):
    '''Fits a model with `train_df[columns]` and returns an
    `EvalResult` with predictions derived from `test_df[columns]`

    :param params: dict of the classifier parameters. It can also be an iterable of
        (key, value) tuples or lists (which will be converted to the {key:value ... }
        corresponding dict)
    '''
    if filepath is not None and isfile(filepath):
        clf = load(filepath)
    else:
        clf = classifier(clf_class, train_df[list(columns)],
                         **{'cache_size': 1500, **dict(params)})
    evres = EvalResult(clf, params, columns)
    if test_df is not None:
        evres.predict(test_df)
    if filepath is not None and not isfile(filepath):
        dump(clf, filepath)
    return evres


class Evaluator:
    '''Creates (and saves) a statisical ML model for outlier detection,
    and launches in parallel a CV
    evaluation, saving all predictions in HDF file, and a summary report in a
    dynamic html page'''

    def __init__(self, clf_class, parameters, rootoutdir=None, n_folds=5):
        '''

        :param parameters: a dict mapping each parameter name (strings) to its list
            of possible values. The total number of cv iterations will be done for all
            possible combinations of all parameters values
        '''
        assert n_folds >= 1
        self.rootoutdir = rootoutdir
        if self.rootoutdir is None:
            self.rootoutdir = abspath(join(dirname(__file__), 'results'))
        if not isdir(self.rootoutdir):
            makedirs(self.rootoutdir)
        if not isdir(self.rootoutdir):
            raise ValueError('Could not create %s' % self.rootoutdir)
        self.clf_class = clf_class
        self.n_folds = n_folds
        # setup self.parameters:
        __p = []
        for pname, vals in parameters.items():
            if not isinstance(vals, (list, tuple)):
                raise TypeError("'%s' must be mapped to a list or tuple of values, "
                                "even when there is only one value to iterate over" %
                                str(pname))
            __p.append(tuple((pname, v) for v in vals))
        self.parameters = tuple(dict(_) for _ in product(*__p))
        self._predictions, self._eval_reports = defaultdict(list), defaultdict(dict)
        self._classes = tuple(_CLASSNAMES)
        # open template file
        with open(join(dirname(__file__), 'eval_report_template.html'), 'r') as _:
            self.eval_report_html_template = _.read()

    def basefilepath(self, *features, **params):
        basepath = abspath(join(self.rootoutdir, self.clf_class.__name__))
        feats, pars = self.tonormtuple(*features), self.tonormtuple(**params)
        suffix = '?' if feats or pars else ''
        if feats:
            suffix += 'features='
            suffix += ','.join(feats)
            if pars:
                suffix += '&'
        if pars:
            suffix += '&'.join('%s=%s' % (k, v) for (k, v) in pars)

        return basepath + suffix

    @staticmethod
    def tonormtuple(*features, **params):
        '''Normalizes features and params into a tuple that is sorted and hashable, so
        that it can be used to uniquely identify the same features and params couple
        '''
        lst = sorted(str(_) for _ in features)
        lst.extend((str(k), str(params[k])) for k in sorted(params))
        return tuple(lst)

    def run(self, dataframe, columns, remove_na=True):
        '''Runs the model evaluation using the data in `dataframe` under the specified
        columns and for all provided parameters
        '''
        print('Running evaluator. All files will be stored in:\n%s' %
              dirname(self.basefilepath()))
        print('with file names prefixed with "%s"' % basename(self.basefilepath()))

        if remove_na:
            print('')
            dataframe = drop_na(dataframe,
                                set(_ for lst in columns for _ in lst),
                                verbose=True).copy()

        # first check: all dataframes are non-empty. This might be due to a set of
        # n-folds for which ...
        for train_df, test_df in self.train_test_split_cv(dataframe):
            err = 'Train' if train_df.empty else 'Test' if test_df.empty else ''
            if err:
                raise ValueError('A %s DataFrame was empty during CV. '
                                 'Try to change the `n_folds` parameter' % err)

        self._predictions.clear()
        self._eval_reports.clear()
        pool = Pool(processes=int(cpu_count()))

        with click.progressbar(length=len(columns) *
                               (1 + self.n_folds) * len(self.parameters)) as pbar:

            def aasync_callback(result):
                self._applyasync_callback(pbar, result)

            for cols in columns:
                # purge the dataframe from duplicates (drop_duplicates)
                # and unnecessary columns (keep_cols). Return a copy at the end
                # of the process. This helps memory mamagement in
                # sub-processes (especialy keep_cols + copy)
                dataframe_ = drop_duplicates(dataframe, cols, 0, False)
                dataframe_ = keep_cols(dataframe_, cols).copy()
                for params in self.parameters:
                    _traindf, _testdf = self.train_test_split_model(dataframe_)
                    fname = self.basefilepath(*cols, **params) + '.model'
                    pool.apply_async(
                        fit_and_predict,
                        (self.clf_class, _traindf, cols, params, _testdf, fname),
                        callback=aasync_callback
                    )
                    for train_df, test_df in self.train_test_split_cv(dataframe_):
                        pool.apply_async(
                            fit_and_predict,
                            (self.clf_class, train_df, cols, params, test_df, None),
                            callback=aasync_callback
                        )

            pool.close()
            pool.join()

    def train_test_split_cv(self, dataframe):
        '''Returns an iterable yielding (train_df, test_df) elements
        for cross-validation. Both DataFrames in each yielded elements are subset
        of `dataframe`
        '''
        return train_test_split(dataframe, self.n_folds)

    def train_test_split_model(self, dataframe):  # pylint: disable=no-self-use
        '''Returns two dataframe representing the train and test dataframe for
        training the global model. Unless subclassed this method returns the tuple:
        ```
        dataframe, None
        ```
        '''
        return dataframe, None

    def _applyasync_callback(self, pbar, eval_result):
        pbar.update(1)

        # make three hashable and sortable objects:
        pkey = self.tonormtuple(**eval_result.params)
        fkey = self.tonormtuple(*eval_result.features)
        fpkey = self.tonormtuple(*eval_result.features, **eval_result.params)

        self._predictions[fpkey].append(eval_result.predictions)  # might be None

        if len(self._predictions[fpkey]) == self.n_folds + 1:
            # finished with predictions of current parameters for the current features,
            # safe as hdf5 and store the summary dataframe of the predicted segments
            # just saved:
            self._eval_reports[fkey][pkey] = \
                self.save_predictions(eval_result.features, eval_result.params)

        sum_dfs = self._eval_reports.get(fkey, {})
        if 0 < len(sum_dfs) == len(self.parameters):  # pylint: disable=len-as-condition
            # finished with the current eval report for the current features,
            # save as csv:
            self.save_evel_report(eval_result.features)

    def save_predictions(self, features, params, delete=True):
        fpkey = self.tonormtuple(*features, **params)
        predicted_df = pdconcat(list(_ for _ in self._predictions[fpkey]
                                     if not getattr(_, 'empty', True)))
        fpath = self.basefilepath(*features, **params)
        predicted_df.to_hdf(
            fpath + '.evalpredictions.hdf', 'cv',
            format='table', mode='w',
            # min_itemsize={'modified': predicted_df.modified.str.len().max()}
        )
        if delete:
            # delete unused data (help gc?):
            del self._predictions[fpkey]
        # now save the summary dataframe of the predicted segments just saved:
        return self.get_summary_df(predicted_df)

    def save_evel_report(self, features, delete=True):
        fkey = self.tonormtuple(*features)
        sum_dfs = self._eval_reports[fkey]

        content = self.eval_report_html_template % {
            'title': self.clf_class.__name__ + " (features: " + ", ".join(fkey) + ")",
            'evaluations': json.dumps([{'key': params,
                                        'data': sumdf.values.tolist()}
                                       for (params, sumdf) in sum_dfs.items()]),
            'columns': json.dumps(next(iter(sum_dfs.values())).columns.tolist()),
            'weights': json.dumps([WEIGHTS[_] for _ in self._classes]),
            'classes': json.dumps(self._classes)
        }

        fpath = self.basefilepath(*features)
        with open(fpath + '.evalreport.html', 'w') as opn_:
            opn_.write(content)
        if delete:
            del self._eval_reports[fkey]

    def get_summary_df(self, predicted_df):
        sum_df_cols = ['ok', 'outlier', '% rec.']

        sum_df = pd.DataFrame(index=self._classes,
                              data=[[0, 0, 0]] * len(self._classes),
                              columns=sum_df_cols,
                              dtype=int)

        for typ, selectorfunc in CLASSES.items():
            _df = predicted_df[selectorfunc(predicted_df)]
            if _df.empty:
                continue
            correctly_pred = _df['correctly_predicted'].sum()
            # map any class defined here to the index of the column above which denotes
            # 'correctly classified'. Basically, map 'ok' to zero and any other class
            # to 1:
            col_idx = 0 if typ == _CLASSNAMES[0] else 1
            # assign value and caluclate percentage recognition:
            sum_df.loc[typ, sum_df_cols[col_idx]] += correctly_pred
            sum_df.loc[typ, sum_df_cols[1-col_idx]] += len(_df) - correctly_pred
            sum_df.loc[typ, sum_df_cols[2]] = \
                np.around(100 * np.true_divide(correctly_pred, len(_df)), 2)

#         oks = sum_df[sum_df_cols[0]]
#         tots = sum_df[sum_df_cols[0]] + sum_df[sum_df_cols[1]]
#         sum_df[sum_col] = np.around(100 * np.true_divide(oks, tots), 2)
        return sum_df
