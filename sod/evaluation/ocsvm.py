'''
Created on 10 Oct 2019

@author: riccardo
'''
from sklearn.svm.classes import OneClassSVM

from sod.evaluation import open_dataset, is_outlier, Evaluator



def run():
    # NOTES: nu=0.9 (also from 0.5) basically shrinks too much and classifies
    # all as outlier
    # All classifiers with delta_pga have lower scores than delta_pga, delta_pgv,
    # thus let's ignore delta_pga
    # All classifiers with delta_pga, psd@10sec have lower scores than
    # delta_pga, delta_pgv, psd@10sec
    # thus let's ignore delta_pga, psd@10sec

    columns = [
        ['delta_pga', 'delta_pgv'],
        ['delta_pga', 'delta_pgv', 'noise_psd@5sec'],
        ['magnitude', 'distance_km', 'amp@0.5hz', 'amp@1hz',
         'amp@2hz', 'amp@5hz', 'amp@10hz'],
        ['magnitude', 'distance_km', 'amp@0.5hz', 'amp@1hz',
         'amp@2hz', 'amp@5hz', 'amp@10hz', 'amp@20hz', 'noise_psd@5sec']
    ]

    parameters = {
        'kernel': ['rbf'],
        'gamma': ['auto', 1, 10, 50],  # np.logspace(np.log10(1), np.log10(50), 4)
        'nu': [0.1, 0.2, 0.5],  # np.logspace(np.log10(.1), np.log10(.5), 3)
    }

    evaluator = OcsvmEvaluator(parameters, n_folds=5)
    evaluator.run(open_dataset(), columns)


class OcsvmEvaluator(Evaluator):

    def __init__(self, parameters, rootoutdir=None, n_folds=5):
        Evaluator.__init__(self, OneClassSVM, parameters, rootoutdir, n_folds)

    def train_test_split_cv(self, dataframe):
        return Evaluator.train_test_split_cv(
            self, dataframe[~is_outlier(dataframe)]
        )

    def train_test_split_model(self, dataframe):
        is_outl = is_outlier(dataframe)
        return dataframe[~is_outl], dataframe[is_outl]

    def run(self, dataframe, columns):
        Evaluator.run(self, dataframe, columns, remove_na=True)


if __name__ == '__main__':
    run()