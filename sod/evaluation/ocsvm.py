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
    cmatrix, df2str, train_test_split, Evaluator
from datetime import datetime, timedelta
from itertools import count, product
from sklearn.svm.classes import OneClassSVM
import os
import pickle
from multiprocessing import Pool, cpu_count


def run():
    dfr_base = open_dataset()

    # NOTES: nu=0.9 (also from 0.5) basically shrinks too much and classifies
    # all as outlier
    # All classifiers with delta_pga have lower scores than delta_pga, delta_pgv,
    # thus let's ignore delta_pga
    # All classifiers with delta_pga, psd@10sec have lower scores than
    # delta_pga, delta_pgv, psd@10sec
    # thus let's ignore delta_pga, psd@10sec

    kolumns = [
        ['delta_pga', 'delta_pgv'],
        ['delta_pga', 'delta_pgv', 'noise_psd@5sec'],
        # ['delta_pga', 'delta_pgv', 'noise_psd@9sec'],
        ['magnitude', 'distance_km', 'amp@0.5hz', 'amp@1hz',
         'amp@2hz', 'amp@5hz', 'amp@10hz'],
        ['magnitude', 'distance_km', 'amp@0.5hz', 'amp@1hz',
         'amp@2hz', 'amp@5hz', 'amp@10hz', 'amp@20hz', 'noise_psd@5sec']
    ]

    all_columns = set(_ for lst in kolumns for _ in lst)
    dfr = dropna(dfr_base, all_columns, verbose=True)

    params = {
        'kernel': ['rbf'],
        'gamma': ['auto', 1, 10, 50],  # np.logspace(np.log10(1), np.log10(50), 4)
        'nu': [0.1, 0.2, 0.5],  # np.logspace(np.log10(.1), np.log10(.5), 3)
    }

    parameters = [{'kernel': k, 'gamma': g, 'nu': n}
                  for k, g, n in product(*list(params.values()))]

    evaluator = OcsvmEvaluator(parameters, n_folds=5)

    for i, columns in enumerate(kolumns, 1):
        print("Step %d of %d" % (i, len(kolumns)))
        print("Features: %s" % str(columns))
        print("")
        evaluator.run(dfr, columns)
        print("")


class OcsvmEvaluator(Evaluator):
    
    def __init__(self, parameters, n_folds=5):
        Evaluator.__init__(self, OneClassSVM, parameters, n_folds)

    def train_test_split(self, dataframe):
        return Evaluator.train_test_split(self,
                                          dataframe[~is_outlier(dataframe)])
    
    def fit_global_model(self, dataframe, columns, **params):
        is_outl = is_outlier(dataframe)
        eval_result = Evaluator.fit_global_model(self,
                                                 dataframe[~is_outl],
                                                 columns,
                                                 **params)
        eval_result.predict(dataframe[is_outl])
        return eval_result


if __name__ == '__main__':
#    print('hello')
    run()
    
    # split_train_dev_test(dfr)