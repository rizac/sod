'''
Implement here an openeer for each dataset created.

Datasets are created via scripts implemented in
`sod.sod.executions`
and saved usually to
`sod.tmp.datasets`

Each file in `sod.tmp.datasets` is usually different, thus we need a way to
open it in such a way. Therefore, after creating a dataset file, you should usually write
here a function **with the same name as the hdf file**, performing the
operations on the input dataframe.

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
from os.path import isfile, isdir, isabs, abspath
import numpy as np
import pandas as pd

from sod.evaluation import capture_stderr, ID_COL, is_prediction_dataframe,\
    is_station_df, dfinfo, normalize


def open_dataset(func):
    def wrapper(filename, normalize_=True, verbose=True):
        if verbose:
            print('Opening %s' % abspath(filename))

        # capture warnings which are redirected to stderr:
        with capture_stderr(verbose):
            dfr = pd.read_hdf(filename)

            if 'Segment.db.id' in dfr.columns:
                if ID_COL in dfr.columns:
                    raise ValueError('The data frame already contains a column '
                                     'named "%s"' % ID_COL)
                # if it's a prediction dataframe, it's for backward compatibility
                dfr.rename(columns={"Segment.db.id": ID_COL}, inplace=True)

            if is_prediction_dataframe(dfr):
                if verbose:
                    print('The dataset contains predictions '
                          'performed on a trained classifier. '
                          'Returning the dataset with no further operation')
                return dfr

            if is_station_df(dfr):
                if verbose:
                    print('The dataset is per-station basis. '
                          'Returning the dataset with no further operation')
                return dfr

            dfr = func(dfr)

            if verbose:
                print('')
                print(dfinfo(dfr))

            if normalize_:
                print('')
                dfr = normalize(dfr)

        # for safety:
        dfr.reset_index(drop=True, inplace=True)
        return dfr

    return wrapper


@open_dataset
def pgapgv(dataframe):
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
            # go to db. We should multuply log * 20 (amp spec) or * 10 (pow spec)
            # but it's unnecessary as we will normalize few lines below
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


@open_dataset
def oneminutewindows(dataframe):
    # save space:
    dataframe['modified'] = dataframe['modified'].astype('category')
    dataframe['window_type'] = dataframe['window_type'].astype('category')
    return dataframe