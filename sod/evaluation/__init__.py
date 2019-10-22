'''
    Evaluation commmon utilities
'''
from os import makedirs
from os.path import abspath, join, dirname, isfile, isdir, basename
from itertools import repeat, product
import warnings
from collections import defaultdict

from joblib import dump, load
import numpy as np
import pandas as pd
import time
from datetime import timedelta
from sklearn.metrics.classification import confusion_matrix
import click
from multiprocessing import Pool, cpu_count
from _io import StringIO
import csv


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
                                'dataset', 'dataset.secondtry.hdf'))


def open_dataset(filename=None, verbose=True):
    if filename is None:
        filename = DATASET_FILENAME
    if verbose:
        print('Opening %s' % filename)
    dfr = pd.read_hdf(filename)
    if verbose:
        print('\nFixing values')

    dfr['delta_pga'] = np.log10(dfr['pga_observed'].abs()) - \
        np.log10(dfr['pga_predicted'].abs())
    dfr['delta_pgv'] = np.log10(dfr['pgv_observed'].abs()) - \
        np.log10(dfr['pgv_predicted'].abs())
    # save space:
    dfr['modified'] = dfr['modified'].astype('category')

    if verbose:
        print(info(dfr))

    sum_df = {}
    if verbose:
        print('\nNormalizing')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        columns = ['min', 'median', 'max', 'NANs', 'segs1-99',
                   'stas1-99']
        oks_ = ~is_outlier(dfr)
        for col in [
            'noise_psd@1sec',
            'noise_psd@2sec',
            'noise_psd@3sec',
            'noise_psd@5sec',
            'noise_psd@9sec',
            'amp@0.5hz',
            'amp@1hz',
            'amp@2hz',
            'amp@5hz',
            'amp@10hz',
            'amp@20hz',
            'magnitude',
            'distance_km',
            'delta_pga',
            'delta_pgv',
            'amplitude_ratio',
            'snr'
        ]:
            if col.startswith('amp@'):
                # go to db. We should multuply log * 20 (amp spec) or * 10 (pow spec)
                # but it's unnecessary as we will normalize few lines below
                dfr[col] = np.log10(dfr[col])
            _dfr = dfr.loc[oks_, :]
            q01 = np.nanquantile(_dfr[col], 0.01)
            q99 = np.nanquantile(_dfr[col], 0.99)
            df1, df99 = _dfr[(_dfr[col] <= q01)], _dfr[(_dfr[col] >= q99)]
            segs1 = len(pd.unique(df1['Segment.db.id']))
            segs99 = len(pd.unique(df99['Segment.db.id']))
            stas1 = len(pd.unique(df1['station_id']))
            stas99 = len(pd.unique(df99['station_id']))

            min_, max_ = np.nanmin(_dfr[col]), np.nanmax(_dfr[col])
            dfr[col] = (dfr[col] - min_) / (max_ - min_)
            if verbose:
                sum_df[col] = {
                    columns[0]: dfr[col].min(),
                    columns[1]:  dfr[col].quantile(0.5),
                    columns[2]: dfr[col].max(),
                    columns[3]: pd.isna(dfr[col]).sum(),
                    columns[4]: segs1 + segs99,
                    columns[5]: stas1 + stas99,
                }
        if verbose:
            print(df2str(pd.DataFrame(data=list(sum_df.values()),
                                      columns=columns,
                                      index=list(sum_df.keys()))))

    if verbose:
        print("-------")
        print("LEGEND:")
        print("%s: unique segments ids (not outliers) "
              "are outside 1 percentile" % columns[4])
        print("%s: unique stations of the segments in %s"
              % (columns[5], columns[4]))
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


def predict(clf, dataframe, *columns):
    '''
    :return: a DataFrame with columns 'label' and 'predicted', where
        both columns values can take wither -1 ('outlier') or 1 ('ok').
    '''
    predicted = _predict(clf, dataframe if not len(columns)
                         else dataframe[list(columns)])
    label = np.ones(len(predicted))
    label[is_outlier(dataframe)] = - 1
    correctly_predicted = label == predicted
    return pd.DataFrame({
        'correctly_predicted': correctly_predicted,
        'outlier': dataframe['outlier'],
        'modified': dataframe['modified'],
        'Segment.db.id': dataframe['Segment.db.id']
    }, index=dataframe.index)


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


def train_test_split(dataframe, n_folds=5):
    dataframe.reset_index(inplace=True, drop=True)
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
    return dfformat(dataframe).to_string()


def dfformat(dataframe):
    strformat = {c: "{:,d}" if str(dataframe[c].dtype).startswith('int')
                 else '{:,.2f}' for c in dataframe.columns}
    return pd.DataFrame({c: dataframe[c].map(strformat[c].format)
                         for c in dataframe.columns},
                        index=dataframe.index)


CLASSES = {
    'ok': lambda dfr: ~is_outlier(dfr),
    'outl. (wrong inv. file)': is_out_wrong_inv,
    'outl. (cha. resp. acc <-> vel)': is_out_swap_acc_vel,
    'outl. (gain X100 or X0.01)': is_out_gain_x100,
    'outl. (gain X10 or X0.1)': is_out_gain_x10,
    'outl. (gain X2 or X0.5)': is_out_gain_x2
}

WEIGHTS = {
    'ok': 100,
    'outl. (wrong inv. file)': 100,
    'outl. (cha. resp. acc <-> vel)': 10,
    'outl. (gain X100 or X0.01)': 50,
    'outl. (gain X10 or X0.1)': 5,
    'outl. (gain X2 or X0.5)': 1
}


def info(dataframe, perclass=True):
    columns = ['segments']
    oks = ~is_outlier(dataframe)
    oks_count = oks.sum()
    if not perclass:
        data = [oks_count, len(dataframe)-oks_count, len(dataframe)]
        index = ['oks', 'outliers', 'total']
    else:
        invfiles = is_out_wrong_inv(dataframe).sum()
        charesp = is_out_swap_acc_vel(dataframe).sum()
        gainx100 = is_out_gain_x100(dataframe).sum()
        gainx10 = is_out_gain_x10(dataframe).sum()
        gainx2 = is_out_gain_x2(dataframe).sum()
        data = [oks_count, invfiles, charesp, gainx100, gainx10, gainx2,
                len(dataframe)]
        index = list(CLASSES.keys()) + ['total']

    return df2str(pd.DataFrame(data, columns=columns, index=index))


class EvalResult:

    __slots__ = ['clf', 'predictions', 'params', 'features']

    def __init__(self, clf, params, features):
        self.clf = clf
        self.params = params
        self.predictions = None
        self.features = tuple(sorted(features))

    def predict(self, dataframe):
        self.predictions = predict(self.clf, dataframe, *self.features)


def fit_and_predict(clf_class, train_df, columns, params, test_df=None,
                    filepath=None):
    '''Fits a model with `train_df[columns]` and returns an
    `EvalResult` with predictions derived from `test_df[columns]`

    :param params: dict of the classifier parameters
    '''
    if filepath is not None and isfile(filepath):
        clf = load(filepath)
    else:
        clf = classifier(clf_class, train_df[list(columns)],
                         **{'cache_size': 2000, **params})
    evres = EvalResult(clf, params, columns)
    if test_df is not None:
        evres.predict(test_df)
    if filepath is not None and not isfile(filepath):
        dump(clf, filepath)
    return evres


class Evaluator:

    def __init__(self, clf_class, parameters, rootoutdir=None, n_folds=5):
        '''
        
        :param parameters: a dict mapping each parameter name (strings) to its list
            of possible values. The total number of cv iterations will be done for all
            possible combinations of all parameters values
        '''
        self.rootoutdir = rootoutdir
        if self.rootoutdir is None:
            self.rootoutdir = abspath(join(dirname(__file__), 'evalresults'))
        if not isdir(self.rootoutdir):
            makedirs(self.rootoutdir)
        if not isdir(self.rootoutdir):
            raise ValueError('Could not create %s' % self.rootoutdir)
        self.clf_class = clf_class
        self.n_folds = n_folds
        # setup self.parameters:
        __p = []
        for p, vals in parameters.items():
            if not isinstance(vals, (list, tuple)):
                raise TypeError("'%s' must be mapped to a list or tuple of values, "
                                "even when there is only one value to iterate over" %
                                str(p))
            __p.append(tuple((p, v) for v in vals))
        self.parameters = tuple(dict(_) for _ in product(*__p))
        self._results = {}
        self._classes = list(CLASSES.keys())
        self.weights = np.array([WEIGHTS[_] for _ in self._classes])

    def basefilepath(self, features, **params):
        basepath = abspath(join(self.rootoutdir, self.clf_class.__name__))
        qry = self.pack2str(features, **params)
        if not qry:
            return basepath
        return basepath + '?' + qry

    def pack2str(self, features=None, **params):
        pars = list((str(k), str(params[k])) for k in sorted(params))
        if features is not None:
            pars.insert(0, ('features', ",".join(str(_) for _ in sorted(features))))
        return '&'.join('%s=%s' % (k, v) for (k, v) in pars)
    
    def unpack(self, query_str):
        ret = {}
        for chunk in query_str.split('&'):
            k, v = chunk.split('=')
            ret[k] = v
        return ret

    def run(self, dataframe, columns):
        self._results = {}
        pool = Pool(processes=cpu_count())
        total = len(columns) * (1 + self.n_folds) * len(self.parameters)

        with click.progressbar(length=total) as pbar:

            def aasync_callback(result):
                self._applyasync_callback(pbar, result)

            aasync = pool.apply_async
            for cols in columns:
                for params in self.parameters:
                    _traindf, _testdf = self.train_test_split(dataframe)
                    fname = self.basefilepath(cols, **params) + '.model'
                    aasync(fit_and_predict,
                           (self.clf_class, _traindf, cols, params, _testdf, fname),
                           callback=aasync_callback)
                    for train_df, test_df in self.train_test_split_cv(dataframe):
                        aasync(fit_and_predict,
                               (self.clf_class, train_df, cols, params, test_df, None),
                               callback=aasync_callback)

            pool.close()
            pool.join()

#         print('Saving evaluation results to file')
#         self.close(columns)
#         print('Done\n')

    def train_test_split_cv(self, dataframe):
        '''Returns an iterable yielding (train_df, test_df) elements
        for cross-validation. Both DataFrames in each yielded elements are subset
        of `dataframe`
        '''
        return train_test_split(dataframe, self.n_folds)

    def train_test_split(self, dataframe):
        '''Returns two dataframe representing the train and test dataframe for
        training the global model. Unless subclassed this method returns the tuple:
        ```
        dataframe, None
        ```
        '''
        return dataframe, None

    def _applyasync_callback(self, pbar, eval_result):
        pbar.update(1)

        evalreport_key = self.pack2str(eval_result.features)
        predictions_key = self.pack2str(eval_result.features, **eval_result.params)

        if predictions_key not in self._results:
            self._results[predictions_key] = []
        self._results[predictions_key].append(eval_result.predictions)

        if len(self._results[predictions_key]) == self.n_folds + 1:
            # finished with predictions of current parameters for the current features,
            # safe as hdf5:
            predicted_df = pdconcat(list(_ for _ in self._results[predictions_key]
                                         if not getattr(_, 'empty', True)))
            fpath = self.basefilepath(eval_result.features, **eval_result.params)
            predicted_df.to_hdf(fpath + '.eval.hdf', 'cv',
                                format='table', mode='w',
                                min_itemsize={'modified': 45})
            # now save the summary dataframe of the predicted segments just saved:
            sum_df = self.get_summary_df(predicted_df)
            if evalreport_key not in self._results:
                self._results[evalreport_key] = {}
            self._results[evalreport_key][self.pack2str(**eval_result.params)] = sum_df
            # delete unused data (help gc?):
            del self._results[predictions_key]

        sum_dfs = self._results.get(evalreport_key, {})
        if 0 < len(sum_dfs) == len(self.parameters):  # pylint: disable=len-as-condition
            # finished with the current eval report for the current features,
            # save as csv:
            _feats = self.unpack(evalreport_key)['features'].replace(',', ';')
            eval_rep = 'Features:;' + _feats + '\n\n'
            eval_rep += self.get_eval_report(sum_dfs)
            fpath = self.basefilepath(eval_result.features)
            with open(fpath + 'eval.allparams.csv', 'w') as opn_:
                opn_.write(eval_rep)
            del self._results[evalreport_key]

    def get_summary_df(self, predicted_df):
        sum_col = '% rec.'
        sum_df_cols = ['ok', 'outlier']
        sum_df = pd.DataFrame(index=self._classes,
                              data=[[0, 0]] * len(self._classes),
                              columns=sum_df_cols,
                              dtype=int)

        for typ, selectorfunc in CLASSES.items():
            _df = predicted_df[selectorfunc(predicted_df)]
            if _df.empty:
                continue
            correctly_pred = _df['correctly_predicted'].sum()
            sum_df.loc[typ, sum_df_cols[0]] += correctly_pred
            sum_df.loc[typ, sum_df_cols[1]] += len(_df) - correctly_pred
        oks = sum_df[sum_df_cols[0]]
        tots = sum_df[sum_df_cols[0]] + sum_df[sum_df_cols[1]]
        sum_df[sum_col] = np.around(100 * np.true_divide(oks, tots), 2)
        return sum_df

    def get_eval_report(self, sum_dfs_dict):
        sum_col = '% rec.'
        stringio = StringIO()
        for params in sorted(sum_dfs_dict):
            sum_df = sum_dfs_dict[params]
            stringio.write('Params:')
            for key, val in self.unpack(params).items():
                stringio.write(';' + str(key))
                stringio.write(';' + str(val))
            stringio.write('\n\n')
            stringio.write(';Classified as:\n')
            dfformat(sum_df).to_csv(stringio, sep=';', quoting=csv.QUOTE_NONE)
            stringio.write(";;Score:;%f\n" %
                           self.get_score(sum_df[sum_col], self.weights))
            stringio.write('\n\n')
        ret = stringio.getvalue()
        stringio.close()
        return ret

    def get_score(self, values, weights):
        res = np.true_divide(np.nansum(values * weights), np.sum(weights))
        return np.around(res, 2)
