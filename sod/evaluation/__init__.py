'''
    Evaluation commmon utilities
'''
from itertools import repeat

import numpy as np
import pandas as pd
import time
from datetime import timedelta


def to_matrix(dataframe, *columns):
    '''Converts dataframe[columns] into a numpy RxC matrix (R=len(dataframe)
    and C=len(columns)) to be fed into OneClassSvm.
    E.g. i.e. for R=3 and C=2, it returns something like (made up numbers):
    [
     [ 2.08392292  1.80265644]
     [ 2.14072406  2.69179574]
     [ 2.43840883  2.22888996]
    ]

    :param dataframe: pandas DataFrame
    :param columns: list of string denoting the columns of `dataframe` that represents
        the feature space to fit the classifier with
    '''
    return (dataframe if not columns else dataframe[list(columns)]).values


def classifier(clf_class, dataframe, *columns, **clf_params):
    '''Returns a OneClassSVM classifier fitted with the data of
    `dataframe`.

    :param dataframe: pandas DataFrame
    :param columns: list of string denoting the columns of `dataframe` that represents
        the feature space to fit the classifier with
    :param clf_params: parameters to be passed to `clf_class`
    '''
    clf = clf_class(**clf_params)
    clf.fit(to_matrix(dataframe, *columns))
    return clf


def predict(clf, dataframe, *columns):
    '''Returns a numpy array of len(dataframe) integers in [-1, 1],
    where:
    -1: item is classified as OUTLIER
     1: item is classified as OK (no outlier)
    Each number at index I is the prediction (classification) of
    the I-th element of `dataframe`

    :param clf: the given (trained) classifier
    :param dataframe: pandas DataFrame
    :param columns: list of string denoting the columns of `dataframe` that represents
        the feature space to fit the classifier with

    '''
    return clf.predict(to_matrix(dataframe, *columns))


# def evaluate(dataframe, predictions, classlabelcol='outlier'):
#     '''
#     :param predicitons: the output of `predict(clf, dataframe, *columns)`
#         for a given classifier `clf` and columns
#     :param classlabelcol: a boolean column denoting if the row is an outlier
#         or not
#     :return: a numpy array with the same values of predictions but where:
#         -1 denotes item misclassified
#         1 denotes item correctly classified
#     '''
#     ret = np.zeros(len(predictions), dtype=int)
#     is_outlier = dataframe[classlabelcol]
#     ret[(is_outlier & (predictions == -1)) |
#         (~is_outlier & (predictions == 1))] = 1
#     ret[ret == 0] = -1
#     return ret
def iterator(n_folds):
    class P:
        def __init__(self, n_split):
            self.t = time.time()
            self.c = 0
            self.n = n_split

        def done(self):
            self.c += 1
            t = time.time() - self.t
            eta = (self.n - self.c) * t / self.c
            print(str(timedelta(seconds=eta)), end='\r')
            time.sleep(1)
    return P(n_folds)


def cross_val_score(clf_class, n_folds, dataframe, *columns, **params):
    '''
        missing doc
    '''
    itr = iterator(n_folds)
    outliers = dataframe['outlier'] != 0
    ok_df = dataframe[~outliers]
    outliers_df = dataframe[outliers]
    assert len(ok_df) + len(outliers_df) == len(dataframe)
    score = 0
    for ok_test_df, outliers_test_df in zip(kfold(ok_df, n_folds, random=True),
                                            kfold(outliers_df, n_folds, random=True)):
        train_df = ok_df[~ok_df.index.isin(ok_test_df.index)]
        clf = classifier(clf_class, train_df, *columns, cache_size=1000, **params)
        dev_df = pdconcat([ok_test_df, outliers_test_df])
        score += get_scores(clf, dev_df, *columns).sum()
        itr.done()
    return score

#     ret.append(dict(params=params, score=score))
#     return sorted(ret, key=lambda item: item['score'], reverse=True)

def get_scores(clf, dataframe, *columns):
    '''
    :return: a numpy array the same length of `dataframe`, with the scores
        obtained by comparing the classes predicted by `clf`, and the
        actual classes. The numbers are weighted according to
        `dataframe['weight']`
    '''
    predictions = predict(clf, dataframe, *columns)
    ret = np.zeros(len(predictions), dtype=float)
    is_outlier = dataframe['outlier'] != 0
    correctly_predicted = \
        (is_outlier & (predictions == -1)) | \
        (~is_outlier & (predictions == 1))
    ret[correctly_predicted] = 1
    ret[~correctly_predicted] = -1
    ret *= dataframe['weight']
    return ret


def pdconcat(dataframes):
    return pd.concat(dataframes, sort=False, axis=0)


def split(size, n_folds):
    step = np.true_divide(size, n_folds)
    for i in range(n_folds):
        yield int(np.ceil(i*step)), int(np.ceil((i+1)*step))


def kfold(dataframe, n_folds, random=True):
    for start, end in split(len(dataframe), n_folds):
        if not random:
            yield dataframe.iloc[start:end]
        else:
            _ = dataframe.copy()
            dfr = _.sample(n=end-start)
            yield dfr
            _ = _[~_.index.isin(dfr.index)]
