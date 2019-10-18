'''
    Evaluation commmon utilities
'''
from os.path import abspath, join, dirname
from itertools import repeat
import warnings

import numpy as np
import pandas as pd
import time
from datetime import timedelta
from sklearn.metrics.classification import confusion_matrix

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
    "psd@0.1sec",
    "psd@1sec",
    "psd@10sec",
    "psd@20sec",
    "psd@50sec",
    "psd@100sec",
    "amp@0.5hz",
    "amp@1hz",
    "amp@2hz",
    "amp@5hz",
    "amp@10hz",
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


def open_dataset(filename=None, verbose=True):
    if filename is None:
        filename = DATASET_FILENAME
    if verbose:
        print('Opening %s' % filename)
    dfr = pd.read_hdf(filename)
    oks = (~is_outlier(dfr)).sum()
    if verbose:
        print(info(dfr))
        print('\nFixing values')

    dfr['delta_pga'] = np.log10(dfr['pga_observed'].abs()) - \
        np.log10(dfr['pga_predicted'].abs())
    dfr['delta_pgv'] = np.log10(dfr['pgv_observed'].abs()) - \
        np.log10(dfr['pgv_predicted'].abs())
    dfr['weight'] = 1
    # save space:
    dfr['modified'] = dfr['modified'].astype('category')

    invfiles = is_out_wrong_inv(dfr)
    charesp = is_out_swap_acc_vel(dfr)
    gainx100 = is_out_gain_x100(dfr)
    gainx10 = is_out_gain_x10(dfr)
    gainx2 = is_out_gain_x2(dfr)
    if verbose:
        print("\nClass (all non 'ok' are outliers) Segments")
        print("ok:                                 %d" % oks)
        print("wrong inventory with file:          %d" % invfiles.sum())
        print("channel response acc <-> vel:       %d" % charesp.sum())
        print("gain x 100:                         %d" % gainx100.sum())
        print("gain x 10:                          %d" % gainx10.sum())
        print("gain x 2:                           %d" % gainx2.sum())

    dfr.loc[invfiles, 'weight'] = 1000
    dfr.loc[charesp, 'weight'] = 100
    dfr.loc[gainx100, 'weight'] = 100
    dfr.loc[gainx10, 'weight'] = 10

    sum_df = {}
    if verbose:
        print('\nNormalizing')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        oks_ = ~is_outlier(dfr)
        for col in [
            'psd@0.1sec',
            'psd@1sec',
            'psd@10sec',
            'psd@20sec',
            'psd@50sec',
            'psd@100sec',
            'amp@0.5hz',
            'amp@1hz',
            'amp@2hz',
            'amp@5hz',
            'amp@10hz',
            'magnitude',
            'distance_km',
            'delta_pga',
            'delta_pgv',
            'amplitude_ratio',
            'snr'
        ]:
            _dfr = dfr.loc[oks_, col]
            min_, max_ = np.nanmin(_dfr), np.nanmax(_dfr)
            dfr[col] = (dfr[col] - min_) / (max_ - min_)
            if verbose:
                sum_df[col] = {
                    'min': dfr[col].min(),
                    'median':  dfr[col].quantile(0.5),
                    'max': dfr[col].max(),
                    'NANs': pd.isna(dfr[col]).sum()
                }
        if verbose:
            print(df2str(pd.DataFrame(data=list(sum_df.values()),
                                      columns=['min', 'median', 'max', 'NANs'],
                                      index=list(sum_df.keys()))))

    return dfr


def drop_duplicates(dataframe, columns, decimals=0, verbose=True):
    dataframe['_t'] = 1
    dataframe.loc[is_out_wrong_inv(dataframe), '_t'] = 2
    dataframe.loc[is_out_swap_acc_vel(dataframe), '_t'] = 3
    dataframe.loc[is_out_gain_x2(dataframe), '_t'] = 4
    dataframe.loc[is_out_gain_x10(dataframe), '_t'] = 5
    dataframe.loc[is_out_gain_x100(dataframe), '_t'] = 6
    if decimals > 0:
        for col in columns:
            dataframe[col] = np.around(dataframe[col], decimals)
    cols = list(columns) + ['_t']
    dataframe2 = dataframe.drop_duplicates(subset=cols)
    del dataframe['_t']
    if verbose:
        print('\nDuplicated per class removed')
        print(info(dataframe2))
    return dataframe2


def dropna(dataframe, columns, verbose=True):
    '''
        Drops rows of dataframe where any column value (at least 1) is NA / NaN

        :return: the new dataframe
    '''
    if verbose:
        print('\nRemoving NA')
    nan_expr = None

    for col in columns:
        expr = pd.isna(dataframe[col])
        nan_count = expr.sum()
        if nan_count == len(dataframe):
            raise ValueError('All values of "%s" NA' % col)
#             if verbose:
#                 print('Removing column "%s" (all NA)' % col)
#             cols2remove.append(col)
        elif nan_count:
            if verbose:
                print('Removing %d NA under column "%s"' % (nan_count, col))
            if nan_expr is None:
                nan_expr = expr
            else:
                nan_expr |= expr

    if nan_expr is not None:
        dataframe = dataframe[~nan_expr].copy()  # pylint: disable=invalid-unary-operand-type
        if verbose:
            print(info(dataframe))

    return dataframe


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


# def iterator(n_folds):
#     class P:
#         def __init__(self, n_split):
#             self.t = time.time()
#             self.c = 0
#             self.n = n_split
#
#         def done(self):
#             self.c += 1
#             t = time.time() - self.t
#             eta = (self.n - self.c) * t / self.c
#             print(str(timedelta(seconds=eta)), end='\r')
#             time.sleep(1)
#     return P(n_folds)


# def cross_val_score(clf_class, n_folds, dataframe, *columns, **params):
#     '''
#         missing doc
#     '''
#     itr = iterator(n_folds)
#     is_outl = is_outlier(dataframe)
#     ok_df = dataframe[~is_outl]
#     outliers_df = dataframe[is_outl]
#     assert len(ok_df) + len(outliers_df) == len(dataframe)
#     score = 0
#     for ok_test_df, outliers_test_df in \
#             zip(kfold(ok_df, n_folds, random=True),
#                 kfold(outliers_df, n_folds, random=True)):
#         train_df = ok_df[~ok_df.index.isin(ok_test_df.index)]
#         clf = classifier(clf_class, train_df, *columns, cache_size=1000,
#                          **params)
#         dev_df = pdconcat([ok_test_df, outliers_test_df])
#         score += get_scores(clf, dev_df, *columns)['score']
#         itr.done()
#     return score


def predict(clf, dataframe, *columns):
    '''
    :return: a DataFrame with columns 'label' and 'predicted', where
        both columns values can take wither -1 ('outlier') or 1 ('ok').
    '''
    predicted = _predict(clf, dataframe if not len(columns)
                         else dataframe[list(columns)])
    label = np.ones(len(predicted))
    is_outl = is_outlier(dataframe)
    label[is_outl] = - 1
    return pd.DataFrame({
        'label': label,
        'predicted': predicted
    }, index=dataframe.index)


def cmatrix(dataframe, sample_weights=None):
    '''
        :param dataframe: the output of predict above
        :param sample_weights: a numpy array the same length of dataframe
            with the sameple weights. Default: None (no weights)

        :return: a pandas 2x2 DataFrame with labels 'ok', 'outlier'
    '''
    labels = ['ok', 'outlier']
    confm = pd.DataFrame(confusion_matrix(dataframe['label'],
                                          dataframe['predicted'],
                                          # labels here just assures that
                                          # 1 (ok) is first and
                                          # -1 (outlier) is second:
                                          labels=[1, -1],
                                          sample_weight=sample_weights),
                         index=labels, columns=labels)
    confm.columns.name = 'Classified as:'
    confm.index.name = 'Label:'
    return confm

#     is_outl = dataframe['label'] == -1
#     predictions =
#     true_positives = (~is_outl & (predictions == 1))
#     false_negatives = (is_outl & (predictions == -1))
#     false_positives = (~is_outl & (predictions == -1))
#     true_negatives = (is_outl & (predictions == 1))
#     correctly_predicted = true_positives | false_negatives
#     ret[correctly_predicted] = 1
#     ret[~correctly_predicted] = -1
#     confm = pd.DataFrame([
#         {'ok': true_positives.sum(), 'outlier': true_negatives.sum()},
#         {'ok': false_positives.sum(), 'outlier': false_negatives.sum()}
#     ], index=['ok', 'outliers'])
#     confm.columns.name = 'Classified as:'
#     confm.index.name = 'Label:'
#     return {
#         'predictions': pd.DataFrame({
#             'Segment.db.id': dataframe['Segment.db.id'],
#             'outlier': dataframe['outlier'],
#             'score': ret}, index=dataframe.index),
#         'cm': confm
#     }


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


def is_outlier(dataframe):
    '''pandas series of boolean telling where dataframe tows are outliers'''
    return dataframe['outlier'] != 0


def is_out_wrong_inv(dataframe):
    return dataframe['modified'].str.contains('INVFILE:')


def is_out_swap_acc_vel(dataframe):
    return dataframe['modified'].str.contains('CHARESP:')


def is_out_gain_x10(dataframe):
    return dataframe['modified'].str.contains('STAGEGAIN:X10.0') | \
        dataframe['modified'].str.contains('STAGEGAIN:X0.1')


def is_out_gain_x100(dataframe):
    return dataframe['modified'].str.contains('STAGEGAIN:X100.0') | \
        dataframe['modified'].str.contains('STAGEGAIN:X0.01')


def is_out_gain_x2(dataframe):
    return dataframe['modified'].str.contains('STAGEGAIN:X2.0') | \
        dataframe['modified'].str.contains('STAGEGAIN:X0.5')


def pdconcat(dataframes):
    '''forwards to pandas concat with standard arguments'''
    return pd.concat(dataframes, sort=False, axis=0, copy=True)


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


def df2str(dataframe):
    strformat = {c: "{:,d}" if str(dataframe[c].dtype).startswith('int')
                 else '{:,.2f}' for c in dataframe.columns}
    newdf = pd.DataFrame({c: dataframe[c].map(strformat[c].format)
                          for c in dataframe.columns},
                         index=dataframe.index)
    return newdf.to_string()


def info(dataframe):
    outliers = is_outlier(dataframe).sum()
    oks = len(dataframe) - outliers
    _str = "%s segments, %s good, %s outliers"
    return (_str % ("{:,d}".format(len(dataframe)),
                    "{:,d}".format(oks),
                    "{:,d}".format(outliers)))


# def to_matri2x(dataframe):
#     '''Converts dataframe[columns] into a numpy RxC matrix (R=len(dataframe)
#     and C=len(columns)) to be fed into OneClassSvm.
#     E.g. i.e. for R=3 and C=2, it returns something like (made up numbers):
#     [
#      [ 2.08392292  1.80265644]
#      [ 2.14072406  2.69179574]
#      [ 2.43840883  2.22888996]
#     ]
# 
#     :param dataframe: pandas DataFrame
#     :param columns: list of string denoting the columns of `dataframe` that represents
#         the feature space to fit the classifier with
#     '''
#     return dataframe.values
# def predi5ct(clf, dataframe, *columns):
#     '''Returns a numpy array of len(dataframe) integers in [-1, 1],
#     where:
#     -1: item is classified as OUTLIER
#      1: item is classified as OK (no outlier)
#     Each number at index I is the prediction (classification) of
#     the I-th element of `dataframe`
# 
#     :param clf: the given (trained) classifier
#     :param dataframe: pandas DataFrame
#     :param columns: list of string denoting the columns of `dataframe` that represents
#         the feature space to fit the classifier with
# 
#     '''
#     return clf.predict(to_matrix(dataframe, *columns))


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
