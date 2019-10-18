'''
Created on 10 Oct 2019

@author: riccardo
'''
from os.path import join, dirname, abspath

import pandas as pd
import numpy as np
from sklearn import svm
from joblib import dump, load
from sod.evaluation import classifier, pdconcat,\
    open_dataset, dropna, is_outlier, info, drop_duplicates, is_out_wrong_inv,\
    is_out_swap_acc_vel, is_out_gain_x100, is_out_gain_x10, is_out_gain_x2, predict,\
    cmatrix, df2str
from datetime import datetime, timedelta
from itertools import count, product
from sklearn.svm.classes import OneClassSVM
import os
import pickle

# dforig = pd.read_hdf(dataset_filename, columns=[])


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
    '''
    '''
    if verbose:
        print('\nSplitting train test')

    dfr.reset_index(inplace=True, drop=True)
    wronginvs = dfr[is_out_wrong_inv(dfr)]
    # test_bad = dfr[is_out_wrong_inv(dfr)]
    # test_ok = dfr[~is_outlier(dfr)].sample(n=len(test_bad))

    _1, _2, _3, _4, _5 = (~is_outlier(dfr),
                          is_out_swap_acc_vel(dfr),
                          is_out_gain_x100(dfr),
                          is_out_gain_x10(dfr),
                          is_out_gain_x2(dfr))

    test_df = pdconcat([
        wronginvs,
        dfr[_1].sample(n=_1.sum() // 3),
        dfr[_2].sample(n=_2.sum() // 3),
        dfr[_3].sample(n=_3.sum() // 3),
        dfr[_4].sample(n=_4.sum() // 3),
        dfr[_5].sample(n=_5.sum() // 3)
    ])
    test_df = test_df.sort_index()
    assert len(pd.unique(test_df.index)) == len(test_df)
    train_df = dfr[~dfr.index.isin(test_df.index)]
    train_df = train_df[~is_outlier(train_df)]

    # FIXME: REMOVE, IT IS JUST FOR PERF REASONS!
    train_df = train_df.sample(n=int(3 * len(train_df) / 6))

    if verbose:
        print('train: %s elements, test: %s elements' %
              ("{:,d}".format(len(train_df)), "{:,d}".format(len(test_df))))

    return train_df, test_df

    # make something simple:
    # take data distributed accoridng to magnitude


def run():
    dfr_base = open_dataset()

    # NOTES: nu=0.9 (also from 0.5) basically shrinks too much and classifies
    # all as outlier

    params = {
        'kernel': ['rbf'],
        'gamma': ['auto', 10, 100],
        'nu': [0.1],  # , 0.9],
        'columns': [['delta_pga'],
                    ['delta_pga', 'delta_pgv'],
                    ['psd@10sec'],
                    ['delta_pga', 'psd@10sec'],
                    ['delta_pga', 'delta_pgv', 'psd@10sec']]
    }

    for kernel, gamma, nu, columns in product(*list(params.values())):
        dfr = dropna(dfr_base, columns, verbose=True)
        dfr = drop_duplicates(dfr, columns, verbose=True)
        train_df, test_df = split_train_test(dfr)
        print('')
        clz = ','.join(_ for _ in columns)
        fle = join(os.getcwd(),
                   "ocsvm_features=%s_kernel=%s_gamma=%s_nu=%s" % (str(clz),
                                                                   str(kernel),
                                                                   str(gamma),
                                                                   str(nu)))
        print('')
        print(os.path.basename(fle))
        if os.path.isfile(fle):
            clf = load(fle)
        else:
            clf = classifier(OneClassSVM, train_df[columns],
                             cache_size=1000, kernel=kernel,
                             gamma=gamma, nu=nu)
            # dump(clf, fle)

        types = {'oks': ~is_outlier(test_df),
                 'wrong inv': is_out_wrong_inv(test_df),
                 'swap acc / vel': is_out_swap_acc_vel(test_df),
                 'gainx100':             is_out_gain_x100(test_df),
                 'gainx10':             is_out_gain_x10(test_df),
                 'gainx2':             is_out_gain_x2(test_df)}

        for typ, selector in types.items():
            print('\nTesting for %s\n' % typ)
            pred_df = predict(clf, test_df[selector], *columns)
            print(df2str(cmatrix(pred_df)))
        print('')
#     scr = cross_val_score(OneClassSVM, 10, train_df, *columns, **params)
#     print(scr)

    print('')

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
#    print('hello')
    run()
    
    # split_train_dev_test(dfr)