'''
Datasets (hdf files) module with IO operations, classes definition and more

Created on 1 Nov 2019

@author: riccardo
'''
import sys
from os.path import (splitext, dirname, join, basename, isfile, isdir, isabs,
                     abspath)
from io import StringIO
from contextlib import contextmanager
import numpy as np
import pandas as pd

from sod.core import odict, CLASSNAMES, CLASS_SELECTORS
from sod.core.paths import DATASETS_DIR



# def is_outlier(dataframe):
#     '''pandas series of boolean telling where dataframe rows are outliers'''
#     return dataframe['outlier']  # simply return the column


##########################
# dataset IO function(s) #
##########################


def open_dataset(filename, categorical_columns=None,
                 verbose=False):
    '''Opens the given dataset located in the dataets directory
    '''
    filepath = dataset_path(filename)
    if verbose:
        title = 'Opening "%s"' % filepath
        print(title)
        print('=' * len(title))

    dataframe = pd.read_hdf(filepath)

    if categorical_columns is not None:
        dfcols = set(dataframe.columns.tolist())
        cols_not_found = []
        for col in categorical_columns:
            if col not in dfcols:
                cols_not_found.append(col)
                continue
            dataframe[col] = dataframe[col].astype('category')

        if verbose and cols_not_found:
            print('WARNING: Skipping the following categorical columns as '
                  'they were not found on the opened dataframe: %s' %
                  cols_not_found)

    if verbose:
        print('Statistics:')
        print('===========')
        print(dfinfo(dataframe))

    return dataframe


def dataset_path(filename, assure_exist=True):
    keyname, ext = splitext(filename)
    if not ext:
        filename += '.hdf'
    filepath = abspath(join(DATASETS_DIR, filename))
    if assure_exist and not isfile(filepath):
        raise ValueError('Invalid dataset, File not found: "%s"'
                         % filepath)
    return filepath


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


def dfnormalize(dataframe, norm_df=None, columns=None, verbose=True):
    '''Normalizes dataframe under the sepcified columns and the specified
    `norm_df` as benchmark dataframe where to take the endpoints (min and max)
    of each column. If `norm_df` is None, it defaults to
    `dataframe` itself. If `columns` is None, it defaults to all floating point
    columns of dataframe.

    :parm norm_df: DataFrame or None. The benchmark dataframe where to take the
        endpoints of each column (min and max) to be normalized. It must have
        the columns specified by `columns`, or the same floating point
        columns of `dataframe`, if `columns` is None or missing. If None,
        defaults to `dataframe`.
    :param columns: if None (the default), nornmalizes on floating columns
        only. Otherwise, it is a list of strings denoting the columns on
        which to normalize
    '''
    if verbose:
        if columns is None:
            print('Normalizing numeric columns (floats only)')
        else:
            print('Normalizing %s' % str(columns))
        print('Normalization is a Rescaling (min-max normalization) where '
              'mina and max are calculated on inliers only'
              'and applied to all instances)')

    if norm_df is None:
        norm_df = dataframe

    with capture_stderr(verbose):
        itercols = floatingcols(dataframe) if columns is None else columns
        for col in itercols:
            # for calculating min and max, we need to drop also infinity, tgus
            # np.nanmin and np.nanmax do not work. Hence:
            finite_values = norm_df[col][np.isfinite(norm_df[col])]
            min_, max_ = np.min(finite_values), np.max(finite_values)
            dataframe[col] = (dataframe[col] - min_) / (max_ - min_)
        if verbose:
            print(dfinfo(dataframe))

    return dataframe


def dfinfo(dataframe, asstring=True):
    '''Returns a a dataframe with info about the given `dataframe` representing
    a given dataset (if asstring=False) or, if asstring=True, a string
    representing the dataframe.
    '''
    classnames = CLASSNAMES
    class_selectors = CLASS_SELECTORS

    sum_dfs = odict()
    for classname, class_selector in zip(classnames, class_selectors):
        _dfr = dataframe[class_selector(dataframe)]
        title = "Class '%s': %d of %d instances" % (classname, len(_dfr),
                                                    len(dataframe))
        sum_dfs[title] = _dfinfo(_dfr)

    # return a MultiIndex DataFrame:
    if not asstring:
        return pd.concat(sum_dfs.values(), axis=0, keys=sum_dfs.keys(),
                         sort=False)

    allstrs = []
    for (key, val) in sum_dfs.items():
        allstrs.extend(['', key, '-' * len(key), val.to_string()])
    return '\n'.join(allstrs)


def _dfinfo(dataframe):
    '''Returns a dataframe with statistical info about the given `dataframe`
    '''
    infocols = ['Min', 'Median', 'Max', '#NAs', '#<1Perc.', '#>99Perc.']
    defaultcolvalues = [np.nan, np.nan, np.nan, 0, 0, 0]
    sum_df = odict()
    # if _dfr.empty
    for col in floatingcols(dataframe):
        colvalues = defaultcolvalues
        if not dataframe.empty:
            q01 = np.nanquantile(dataframe[col], 0.01)
            q99 = np.nanquantile(dataframe[col], 0.99)
            df1, df99 = dataframe[(dataframe[col] < q01)], dataframe[(dataframe[col] > q99)]
            colvalues = [
                np.nanmin(dataframe[col]),  # Min
                np.nanquantile(dataframe[col], 0.5),  # Median
                np.nanmax(dataframe[col]),  # Max
                (~np.isfinite(dataframe[col])).sum(),  # #NAs
                len(df1),  # #<1Perc.
                len(df99)  # @>99Perc.
            ]

        sum_df[col] = {i: v for i, v in zip(infocols, colvalues)}

    return pd.DataFrame(data=list(sum_df.values()),
                        columns=infocols,
                        index=list(sum_df.keys()))


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


def floatingcols(dataframe):
    '''Iterable yielding all floating point columns of dataframe'''
    for col in dataframe.columns:
        try:
            if np.issubdtype(dataframe[col].dtype, np.floating):
                yield col
        except TypeError:
            # categorical data falls here
            continue
