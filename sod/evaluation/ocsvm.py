'''
Created on 10 Oct 2019

@author: riccardo
'''
from os.path import join, dirname, abspath

import pandas as pd
import numpy as np
from sklearn import svm
from sod.evaluation import classifier, predict, kfold, pdconcat,\
    cross_val_score, get_scores
from datetime import datetime, timedelta
from itertools import count
from sklearn.svm.classes import OneClassSVM

dataset_filename = join(dirname(__file__), '..', 'dataset', 'dataset.hdf')

# dforig = pd.read_hdf(dataset_filename, columns=[])

def open_dataset(filename):
    dfr = pd.read_hdf(filename)
    dfr['delta_pga'] = np.log10(dfr['pga_observed'].abs()) - \
        np.log10(dfr['pga_predicted'].abs())  # / dforig['pga_predicted']#.abs()
    dfr['delta_pgv'] = np.log10(dfr['pgv_observed'].abs()) - \
        np.log10(dfr['pgv_predicted'].abs())  # / dforig['pgv_predicted']#.abs()
    dfr['weight'] = 1
    dfr['modified'] = dfr['modified'].astype('category')
    dfr.loc[dfr['modified'].str.contains('INVFILE:'), 'weight'] = 1000
    dfr.loc[dfr['modified'].str.contains('CHARESP:'), 'weight'] = 100
    dfr.loc[dfr['modified'].str.contains('STAGEGAIN:X10.0') |
            dfr['modified'].str.contains('STAGEGAIN:X0.1'), 'weight'] = 10
    dfr.loc[dfr['modified'].str.contains('STAGEGAIN:X100.0') |
            dfr['modified'].str.contains('STAGEGAIN:X0.01'), 'weight'] = 100

    dfr = dfr[~(dfr['modified'].str.contains('STAGEGAIN:X2.0') | 
                dfr['modified'].str.contains('STAGEGAIN:X0.5'))]
#     dfr = dfr.drop(dfr['modified'].str.contains('STAGEGAIN:X2.0') | 
#              dfr['modified'].str.contains('STAGEGAIN:X0.5'))

#     dfr['mag_group'] = np.round(dfr['magnitude'], 0).astype(int)
#     dfr['dist_group'] = np.log10(dfr['distance_km']).astype(int)
    return dfr.copy()

# def test_open_dataset(filename):
#     # TO BE REMOVED WHEN WE HAVE THE NEW DATASET!
#     dfr = open_dataset(filename)
#     dfr.drop(['saturated', 'low_snr'], axis=1, inplace=True)
#     dtimes = [datetime(1990, 1, 1) + timedelta(days=.6*_) for _ in range(len(dfr))]
#     dfr['event_time'] = dtimes
#     staids = dtimes = [d.month for d in dtimes]
#     dfr['sta_id'] = staids
#     fname = abspath(join(dirname(filename), 'dataset.tmp.hdf'))
#     dfr.to_hdf(fname, 's2s_table')


def ocsvm_fit(dataframe, *columns, **params):
    return classifier(svm.OneClassSVM,
                      dataframe,
                      *columns,
                      **params)


iterations_parameters = {
    'kernel': ['rbf'],
    'gamma': ['auto', 1, 10, 100],
    'nu': [0.1, 0.5, .9]
}


def split_train_test(dfr, verbose=True):

    if verbose:
        outliers = dfr['outlier'].sum()
        print("%d segments, %d good, %d outliers" %
              (len(dfr), len(dfr)-outliers, outliers))

    test_bad = dfr[dfr['modified'].str.contains('INV')]
    test_ok = dfr[dfr['outlier'] == 0].sample(n=len(test_bad))
    test_df = pdconcat([test_bad, test_ok])

    train_df = dfr[~dfr.index.isin(test_df.index)]

    if verbose:
        print('train: %d elements, test: %d elements' %
              (len(train_df), len(test_df)))

    return train_df, test_df

    # make something simple:
    # take data distributed accoridng to magnitude

def run():
    dfr = open_dataset(dataset_filename)
    dfr = dfr[dfr['snr'] >= 3]
    train_df, test_df = split_train_test(dfr)
    params = {'kernel': 'rbf', 'gamma': 'auto', 'nu': 0.9}
    columns = ['delta_pga', 'delta_pgv']
#     scr = cross_val_score(OneClassSVM, 10, train_df, *columns, **params)
#     print(scr)
    scores = get_scores(classifier(OneClassSVM,
                                   train_df[train_df['outlier'] == 0],
                                   *columns,
                                   cache_size=1000, **params),
                        test_df, *columns)
    print("%f %d %d" % (scores.sum(), (scores > 0).sum(), (scores < 0).sum()))
#     _ds  = pd.cut(dfr['distance_km'], np.array([0, 10000]), right=False)
#     assert not pd.isna(_mg).any()
#     assert not pd.isna(_ds).any()
#     dfr_ = dfr.groupby([_mg, _ds]).size().unstack().fillna(0).astype(int)
#     print(dfr_)
#     assert dfr_.sum().sum() == len(dfr)
#     asd = 9

    # split by magnitudes
#     _ev_years = np.unique(dfr['event_time'].dt.year)
#     in_boundary_years = (dfr['event_time'].dt.year == _ev_years[0]) |
#                  (dfr['event_time'].dt.year == _ev_years[-1])
#     _good_df_in_boundary_years = dfr[in_boundary_years & df['outlier'] == 0]
#     np.random.randint(2, size=10)
#     
#     print(ev_years)
#     dev_df = dfr[(dfr['event_time'].dt.year == ev_years[0]) |
#                  (dfr['event_time'].dt.year == ev_years[-1])]
#     print('dev: %d' % len(dev_df))
#     
#     test_df2 = dfr[(dfr['outlier']==True) & (~dfr['modified'].str.contains('INV'))]
#     print('test2: %d' % len(test_df2))


if __name__ == '__main__':
    run()
    
    # split_train_dev_test(dfr)