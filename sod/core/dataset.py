'''
Implement here an openeer for each dataset created.

Datasets are created via scripts implemented in
`sod.sod.executions`
and saved usually to
`sod.tmp.datasets`

Each file in `sod.tmp.datasets` is usually different, thus we need a way to
open it in such a way. Therefore, after creating a dataset file, you should
usually write here a function **with the same name as the hdf file**,
performing the operations on the input dataframe.

The method **must be decorated** with `open_dataset`, making the decorated
function with signature:
`func(filename=None, normalize_=True, verbose=True)`

The decorated function must be
implemented with a different signature:
`func(dataframe)`
and will be called by the decorator after reading the hdf file and prior
to normalization (if the function will be called with normalize_ = True)

Created on 1 Nov 2019

@author: riccardo
'''
import sys
from os.path import splitext, dirname, join, basename
from io import StringIO
from contextlib import contextmanager
from os.path import isfile, isdir, isabs, abspath
import numpy as np
import pandas as pd

from sod.core.evaluation import (ID_COL, is_prediction_dataframe,
                                 is_outlier, CLASSES, pdconcat)
from sod.core.paths import DATASETS_DIR


def dataset_path(filename, assure_exist=True):
    keyname, ext = splitext(filename)
    if not ext:
        filename += '.hdf'
    filepath = join(DATASETS_DIR, filename)
    if assure_exist and not isfile(filename):
        raise ValueError('Invalid dataset, no file found with name "%s"'
                         % keyname)
    return filepath


def open_dataset(filename, normalize=True, verbose=True):

    filepath = dataset_path(filename)
    keyname = splitext(basename(filepath))[0]

    try:
        func = globals()[keyname]
    except KeyError:
        raise ValueError('Invalid dataset, no function "%s" '
                         'implemented' % keyname)

    if verbose:
        print('Opening %s' % abspath(filepath))

    # capture warnings which are redirected to stderr:
    with capture_stderr(verbose):
        dfr = pd.read_hdf(filepath)

        if 'Segment.db.id' in dfr.columns:
            if ID_COL in dfr.columns:
                raise ValueError('The data frame already contains a '
                                 'column named "%s"' % ID_COL)
            # if it's a prediction dataframe, it's for backward compatib.
            dfr.rename(columns={"Segment.db.id": ID_COL}, inplace=True)

        try:
            dfr = func(dfr)
        except Exception as exc:
            raise ValueError('Check module function "%s", error: %s' %
                             (func.__name__, str(exc)))

        if verbose:
            print('')
            print(dfinfo(dfr))

        if normalize:
            print('')
            dfr = dfnormalize(dfr, None, verbose)

    # for safety:
    dfr.reset_index(drop=True, inplace=True)
    return dfr


#################
# Functions mapped to specific datasets in 'datasets' and performing
# custom dataframe operations
#################


def pgapgv(dataframe):
    '''Custom operations to be performed on the pgapgv dataset
    (sod/datasets/pgapgv.hdf)
    '''
    # setting up columns:
    dataframe['pga'] = np.log10(dataframe['pga_observed'].abs())
    dataframe['pgv'] = np.log10(dataframe['pgv_observed'].abs())
    dataframe['delta_pga'] = np.log10(dataframe['pga_observed'].abs()) - \
        np.log10(dataframe['pga_predicted'].abs())
    dataframe['delta_pgv'] = np.log10(dataframe['pgv_observed'].abs()) - \
        np.log10(dataframe['pgv_predicted'].abs())
    del dataframe['pga_observed']
    del dataframe['pga_predicted']
    del dataframe['pgv_observed']
    del dataframe['pgv_predicted']
    for col in dataframe.columns:
        if col.startswith('amp@'):
            # go to db. We should multuply log * 20 (amp spec) or * 10 (pow
            # spec) but it's unnecessary as we will normalize few lines below
            dataframe[col] = np.log10(dataframe[col])
    # save space:
    dataframe['modified'] = dataframe['modified'].astype('category')
    # numpy int64 for just zeros and ones is waste of space: use bools
    # (int8). But first, let's be paranoid first (check later, see below)
    _zum = dataframe['outlier'].sum()
    # convert:
    dataframe['outlier'] = dataframe['outlier'].astype(bool)
    # check:
    if dataframe['outlier'].sum() != _zum:
        raise ValueError('The column "outlier" is supposed to be '
                         'populated with zeros or ones, but conversion '
                         'to boolean failed. Check the column')
    return dataframe


def oneminutewindows(dataframe):
    '''Custom operations to be performed on the oneminutewindows dataset
    (sod/datasets/oneminutewindows.hdf)
    '''
    # save space:
    dataframe['modified'] = dataframe['modified'].astype('category')
    dataframe['window_type'] = dataframe['window_type'].astype('category')
    return dataframe


def magnitudeenergy(dataframe):
    '''Custom operations to be performed on the magnitudeenergy dataset
    (sod/datasets/magnitudeenergy.hdf)
    '''
    # save space:
    dataframe['modified'] = dataframe['modified'].astype('category')
    return dataframe


###########################
# Other operations
###########################


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
    if isinstance(syserr, StringIO):
        # already within a captured_stderr with statement?
        yield
    else:
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


def dfinfo(dataframe, perclass=True):
    '''Returns a adataframe with info about the given `dataframe` representing
    a given dataset
    '''
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


def df2str(dataframe):
    ''':return: the string representation of `dataframe`, with numeric values
    formatted with comma as decimal separator
    '''
    return _dfformat(dataframe).to_string()


def _dfformat(dataframe, n_decimals=2):
    '''Returns a copy of `dataframe` with all numeric values converted to
    formatted strings (with comma as thousand separator)

    :param n_decimals: how many decimals to display for floats (defautls to 2)
    '''
    float_frmt = '{:,.' + str(n_decimals) + 'f}'  # e.g.: '{:,.2f}'
    strformat = {
        c: "{:,d}" if str(dataframe[c].dtype).startswith('int') else float_frmt
        for c in dataframe.columns
    }
    return pd.DataFrame({c: dataframe[c].map(strformat[c].format)
                         for c in dataframe.columns},
                        index=dataframe.index)


def dfnormalize(dataframe, columns=None, verbose=True):
    '''Normalizes dataframe under the sepcified columns. Only good instances
    (not outliers) will be considered in the normalization

    :param columns: if None (the default), nornmalizes on floating columns
        only. Otherwise, it is a list of strings denoting the columns on
        which to normalize
    '''
    sum_df = {}
    if verbose:
        if columns is None:
            print('Normalizing numeric columns (floats only)')
        else:
            print('Normalizing %s' % str(columns))
        print('(only good instances - no outliers - taken into account)')

    with capture_stderr(verbose):
        infocols = ['prenorm_min', 'prenorm_max',
                    'min', 'median', 'max', 'NAs', 'ids outside [1-99]%']
        oks_ = ~is_outlier(dataframe)
        itercols = floatingcols(dataframe) if columns is None else columns
        for col in itercols:
            _dfr = dataframe.loc[oks_, :]
            q01 = np.nanquantile(_dfr[col], 0.01)
            q99 = np.nanquantile(_dfr[col], 0.99)
            df1, df99 = _dfr[(_dfr[col] < q01)], _dfr[(_dfr[col] > q99)]
            segs1 = len(pd.unique(df1[ID_COL]))
            segs99 = len(pd.unique(df99[ID_COL]))
            # stas1 = len(pd.unique(df1['station_id']))
            # stas99 = len(pd.unique(df99['station_id']))

            # for calculating min and max, we need to drop also infinity, tgus
            # np.nanmin and np.nanmax do not work. Hence:
            finite_values = _dfr[col][np.isfinite(_dfr[col])]
            min_, max_ = np.min(finite_values), np.max(finite_values)
            dataframe[col] = (dataframe[col] - min_) / (max_ - min_)
            if verbose:
                sum_df[col] = {
                    infocols[0]: min_,
                    infocols[1]: max_,
                    infocols[2]: dataframe[col].min(),
                    infocols[3]: dataframe[col].quantile(0.5),
                    infocols[4]: dataframe[col].max(),
                    infocols[5]: (~np.isfinite(dataframe[col])).sum(),
                    infocols[6]: segs1 + segs99,
                    # columns[5]: stas1 + stas99,
                }
        if verbose:
            print(df2str(pd.DataFrame(data=list(sum_df.values()),
                                      columns=infocols,
                                      index=list(sum_df.keys()))))
            print("-------")
            print("Min and max might be outside [0, 1]: the normalization ")
            print("bounds are calculated on good segments (non outlier) only")
            print("%s: values which are NaN or Infinity" % infocols[5])
            print("%s: good instances (not outliers) the given percentiles" %
                  infocols[6])

    return dataframe


def floatingcols(dataframe):
    '''Iterable yielding all floating point columns of dataframe'''
    for col in dataframe.columns:
        try:
            if np.issubdtype(dataframe[col].dtype, np.floating):
                yield col
        except TypeError:
            # categorical data falls here
            continue


####################
# TO BE TESTED!!!!
####################


NUM_SEGMENTS_COL = 'num_segments'


def is_station_df(dataframe):
    '''Returns whether the given dataframe is the result of `groupby_station`
    on a given segment-based dataframe
    '''
    return NUM_SEGMENTS_COL in dataframe.columns


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
            _dfmedian = _df[fl_cols].median(axis=0, numeric_only=True,
                                            skipna=True)
            _dfmedian[NUM_SEGMENTS_COL] = len(_df)
            _dfmedian['outlier'] = outlier
            _dfmedian['modified'] = modified
            _dfmedian[ID_COL] = staid
            newdf.append(pd.DataFrame([_dfmedian]))
            # print(pd.DataFrame([_dfmedian]))

        ret = pdconcat(newdf, ignore_index=True)
        ret[NUM_SEGMENTS_COL] = ret[NUM_SEGMENTS_COL].astype(int)
        # convert dtypes because they might not match:
        shared_c = (set(dataframe.columns) & set(ret.columns)) - set(fl_cols)
        for col in shared_c:
            ret[col] = ret[col].astype(dataframe[col].dtype)
        if verbose:
            bins = [1, 10, 100, 1000, 10000]
            max_num_segs = ret[NUM_SEGMENTS_COL].max()
            if max_num_segs >= 10 * bins[-1]:
                bins.append(max_num_segs + 1)
            elif max_num_segs >= bins[-1]:
                bins[-1] = max_num_segs + 1
            groups = ret.groupby(pd.cut(ret[NUM_SEGMENTS_COL], bins,
                                        precision=0,
                                        right=False))
            print(pd.DataFrame(groups.size(), columns=['num_stations']).
                  to_string())
            assert groups.size().sum() == len(ret)
            print('')
            print('Summary of the new dataset (instances = stations)')
            print(dfinfo(ret))
        return ret


