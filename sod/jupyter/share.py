'''
`share.py`: shared stuff for my notebooks. This module should be used only from
within a Jupyter Notebook.

---
When using the module `share` in a Jupyter Notebook, note the following naming
conventions:

`eval_df` denotes an Evaluation Data frame, i.e. a
tabular object where *a row is a model evaluation* and columns report several
evaluation info, e.g., model hyperparams, testset file path,
evaluation metrics scores.

`pred_df` denotes a Prediction Data frame representing an evaluation
(= row of `eval_df`) in details : *a row is a testset instance* and columns
report several instance info, including the instance actual class as boolean
(column 'outlier') and the predicted class/score as float in [0, 1] (column
'predicted_anomaly_score')
---

Both dataframes are the output of several experiments run via
the script `evaluate.py`

Created on 18 Mar 2020

@author: rizac(at)gfz-potsdam.de
'''
from os.path import join, abspath, dirname, isfile, isdir
import sys
import os
import re
import time
import inspect
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, namedtuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hmean
from sklearn import metrics
from joblib import dump, load
from IPython.display import display_html, clear_output  # https://stackoverflow.com/a/36313217
import pandas as pd
import contextlib
# for printing, we can do this:
# with pd.option_context('display.max_rows', -1, 'display.max_columns', 5):
# or we simply set once here the max_col_width
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.max_columns', 500)


# dump.__doc__ = ('`dump(value, filename)` persists an arbitrary Python object '
#                 'into one file. For details, see `joblib.dump`')
# load.__doc__ = ('`load(filepath)`. Reconstruct a Python object from a file '
#                 'persisted with `dump`. For details, see `joblib.load`')


def printhtml(what):
    '''Same as display_html(text, True): prints the html formatted text
    passed as argument'''
    display_html(what, raw=True)


# EVALPATH = join(dirname(dirname(__file__)), 'evaluations', 'results')
# assert isdir(EVALPATH)


class EVALMETRICS(Enum):
    '''Evaluation metrics enumeration. Each enum item is mapped to a string:
    `AUC` = 'roc_auc_score'
    `APS` = 'average_precision_score'
    `LOGLOSS` = 'log_loss'
    `F1MAX` = 'f1_max'
     and has a `compute` method
    (e.g., `EVALMETRICS.APS.compute(pred_df)`) which
    returns the metric scalar value from a prediction dataframe
    '''
    AUC = 'roc_auc_score'
    APS = 'average_precision_score'
    LOGLOSS = 'log_loss'
    F1MAX = 'f1_max'

    def __str__(self):
        return self.value

    def compute(self, pred_df):
        '''Computes the value of this metric'''
        y_true, y_pred = pred_df.outlier, pred_df.predicted_anomaly_score
        if self == EVALMETRICS.AUC:
            return metrics.roc_auc_score(y_true, y_pred)
        if self == EVALMETRICS.APS:
            return metrics.average_precision_score(y_true, y_pred)
        if self == EVALMETRICS.F1MAX:
            pre, rec, thr = metrics.precision_recall_curve(y_true, y_pred)
            fscores = f1scores(pre, rec)
            argmax = np.argmax(fscores)
            return fscores[argmax]
        if self == EVALMETRICS.LOGLOSS:
            return metrics.log_loss(y_true, y_pred)
        raise ValueError('something wrong in EVALMETRIS.compute')

# EVAL_SCORES = ['roc_auc_score', 'average_precision_score', 'log_loss']


# class HYPERPARAMS(Enum):
#     '''Model hyper parameters. Each item is mapped to a string:
#     `NEST` = 'n_estimators'
#     `MAXS` = 'max_samples'
#     `RSTATE` = 'random_state'
#     '''
#     NEST = 'n_estimators'
#     MAXS = 'max_samples'
#     RSTATE = 'random_state'
#
#     def __str__(self):
#         return self.value


# _evalcolumns = ['clf', 'features', 'n_estimators', 'max_samples',
#                 'random_state', 'contamination', 'behaviour',
#                 'tr_set', 'filepath']
# 
# Evaluation = namedtuple('Evaluation', _evalcolumns)


# class EVALCOLUMNS(Enum):
#     '''Evaluation dataframe columns mapped to a string:
#     `NEST` = 'n_estimators'
#     `MAXS` = 'max_samples'
#     `RSTATE` = 'random_state'
#     '''
#     FEATS = 'feats'
#     NEST = 'n_estimators'
#     MAXS = 'max_samples'
#     RSTATE = 'random_state'
#     FILEPATH = 'filepath'
# 
#     def __str__(self):
#         return self.value
# 
#     @classmethod
#     def hyperparams(cls):
#         return [cls.NEST.value, cls.MAXS.value, cls.RSTATE.value]


def read_summary_eval_df(**kwargs):
    '''`read_summary_eval_df(**kwargs)` = pandas `read_hdf(EVALPATH, **kwargs)`
    reads and returns the Evaluation dataframe created and incremented by each
    execution of the main script `evaluate.py`. *Thus, if no
    evaluation has been run on this computer, no evaluation dataframe exists
    and the Notebook using this module will not work.*
    Arguments of the functions are the same keyword arguments as pandas
    `read_hdf` (only the file path must not be given because hard coded
    and relative to the 'evaluations/results' subfolder of this package).
    '''
    dfr = pd.read_hdf(_abspath('summary_evaluationmetrics.hdf'), **kwargs)
    # _key is the prediction dataframe path, relative to
    # the EVALPATH directory I guess. Create a new filepath column with
    # the complete path of each prediction:
    # fpathkey = 'filerelativepath'
    dfr.rename(columns={'_key': 'file_relative_path'}, inplace=True)
#     dfr[fpathkey] = EVALPATH
#     dfr[fpathkey] = dfr[fpathkey].str.cat(dfr['_key'], sep=os.sep)
#     dfr.drop('_key', axis=1, inplace=True)
    # re-order columns to make them more readable. Define the
    # columns to appear first (columns describing the model, e.g. hyperparams)
    # and then important evaluation metrics:
    colorder = [
        'clf', 'feats', 'n_estimators', 'max_samples', 'random_state',
        str(EVALMETRICS.AUC), str(EVALMETRICS.APS), str(EVALMETRICS.LOGLOSS)
    ]
    return dfr[colorder +
               sorted(_ for _ in dfr.columns if _ not in colorder)].copy()


def _abspath(evalresult_relpath):
    '''Returns the absolute path of `evalresult_relpath` which is supposed
    to be relative of the evaluation result directory of this package.
    The latter exists only if some evaluation has
    been run
    '''
    EVALPATH = join(dirname(dirname(__file__)), 'evaluations', 'results')
    return abspath(join(EVALPATH, evalresult_relpath))


# check:

if not isdir(_abspath('')):
    raise ValueError('The evaluation directory and could not be found. '
                     'Have you run some evaluations on this machine '
                     '(script `evaluate.py`)?')


# PLOTTING RELATED STUFF:


def samex(axes):
    '''`samex(axes)` sets the same x limits on all matplotlib Axes provided
    as argument'''
    return _sameaxis(axes, 'x')


def samey(axes):
    '''`samey(axes)` sets the same x limits on all matplotlib Axes provided
    as argument'''
    return _sameaxis(axes, 'y')


def _sameaxis(axes, meth):
    if axes is None or not len(axes):  # pylint: disable=len-as-condition
        return None, None
    lims = np.array([_.get_xlim() for _ in axes]) if meth == 'x' else \
        np.array([_.get_ylim() for _ in axes])
    lmin, lmax = np.nanmin(lims[:, 0]), np.nanmax(lims[:, 0])
    for ax_ in axes:
        if meth == 'x':
            ax_.set_xlim(lmin, lmax)
        else:
            ax_.set_ylim(lmin, lmax)
    return lmin, lmax


def plot_feats_vs_evalmetrics(eval_df, evalmetrics=None, show=True):
    '''`plot_feats_vs_em(eval_df, evalmetrics=None, show=True)` plots the
    features of `eval_df` grouped and
    colored by PSD period counts versus the given `ems` (Evaluation metrics,
    passes as strings or `EVALMETRICS` enum items. None - the default -
    shows all possile metrics stored in a `eval_df`: Area under ROC curve,
    Average precision score, log loss)
    '''
    if evalmetrics is None:
        evalmetrics = [EVALMETRICS.AUC, EVALMETRICS.APS, EVALMETRICS.LOGLOSS]
    feats = features(eval_df)
    feat_labels = [
        _.replace('psd@', '').replace('sec', '').replace(',', ' ')
        for _ in feats
    ]

    colors = get_colors(max(len(_.split(',')) for _ in feats))
    fig = plt.figure(constrained_layout=True)
    gsp = fig.add_gridspec(1, len(evalmetrics))

    for j, metric_name in enumerate(str(_) for _ in evalmetrics):
        axs = fig.add_subplot(gsp[0, j])
        minx, maxx = None, None
        for i, feat in enumerate(feats):
            df_ = eval_df[eval_df.feats == feat][metric_name]
            min_, median, max_ = df_.min(), df_.median(), df_.max()

            xerr = [[median-min_], [max_-median]]
            color = colors[len(feat.split(',')) - 1]

            # print errbar background:
            axs.barh(i, left=min_, width=max_-min_, height=.8, alpha=0.25,
                     color=color, linewidth=0)
            # print errbar border:
            # ax.barh(i, left=min_, width=max_-min_, height=0.75, alpha=1,
            #         fill=False, ec=color, linewidth=1)

            axs.errorbar(median, i, xerr=xerr, color=color, marker='|',
                         capsize=0, linewidth=0, elinewidth=0, capthick=0,
                         markersize=20, mew=2)

            # don't know why axes does not set automatically xlim, maybe
            # barh is not working as expected?
            if i == 0:
                minx, maxx = min_, max_
            else:
                minx, maxx = min(minx, min_), max(maxx, max_)

        margin = (maxx - minx) * 0.05
        axs.set_xlim(minx - margin/2, maxx + margin/2)

        axs.set_yticks(list(range(len(feats))))
        if j == 0:
            axs.set_ylabel('Features (PSD periods)')
            axs.set_yticklabels(feat_labels)
        else:
            axs.set_yticklabels([])

        axs.set_xlabel(metric_name.replace('_', ' '))
        axs.grid()

    if show:
        plt.show()
        fig = None

    return fig


_DEFAULT_COLORMAP = 'cubehelix'


@contextlib.contextmanager
def use_tmp_colormap(name):
    '''`with use_tmp_colormap(name)` changes temporarily the colormap. Useful
    before plotting
    '''
    global _DEFAULT_COLORMAP
    _ = _DEFAULT_COLORMAP
    _DEFAULT_COLORMAP = name
    try:
        yield
    finally:
        _DEFAULT_COLORMAP = _


def get_colors(numcolors):
    cmap = plt.get_cmap(_DEFAULT_COLORMAP)
    # numcolors+2 makes a margin in order to avoid extreme colors:
    return [cmap(i, 1) for i in np.linspace(0, 1, numcolors+2, endpoint=True)]


def features(eval_df):
    '''`features(eval_df)` returns a list of sorted strings representing
    the unique features of the DataFrame passed as argument. Features are
    sorted by number of PSD periods and if equal, by periods sum
    '''
    feats = pd.unique(eval_df.feats)

    def sortfunc(feats):
        psds = []
        for feat in feats.split(','):
            psds.append(float(feat.replace('psd@', '').replace('sec', '')))
        return np.sum([(10 ** i) * p for i, p in enumerate(psds)])
    return sorted(feats, key=sortfunc)


def get_hyperparam_dfs(eval_df, evalmetric, **hyperparams):
    '''`get_hyperparam_dfs(eval_df, evalmetric, hparam1=values, hparam2=values)`
    returns the three dataframes min, median max where the axis are the two
    hyperparameters values and the cell value is the score min, median and max
    '''
    hp1 = str(list(hyperparams.keys())[0])
    hp2 = str(list(hyperparams.keys())[1])
    hp1values = hyperparams[hp1]
    hp2values = hyperparams[hp2]

    index = pd.MultiIndex.from_tuples([(hp1, _) for _ in hp1values])
    columns = pd.MultiIndex.from_tuples([(hp2, _) for _ in hp2values])

    # df_hmean = pd.DataFrame(data=0, index=n_estim, columns=max_samp)
    df_median = pd.DataFrame(data=0, index=index, columns=columns)
    df_max = pd.DataFrame(data=0, index=index, columns=columns)
    df_min = pd.DataFrame(data=0, index=index, columns=columns)
    # df_var = pd.DataFrame(data=0, index=index, columns=columns)

    for hp1val in hp1values:
        for hp2val in hp2values:
            dfr = eval_df[(eval_df[hp1] == hp1val) & (eval_df[hp2] == hp2val)]
            vals = dfr[str(evalmetric)]
            df_median.loc[(hp1, hp1val), (hp2, hp2val)] = vals.median()
            df_max.loc[(hp1, hp1val), (hp2, hp2val)] = vals.max()
            df_min.loc[(hp1, hp1val), (hp2, hp2val)] = vals.min()

    return df_min, df_median, df_max


def plot_hyperparam_dfs(df_min, df_median, df_max, ylabel=None, show=True):
    '''`plot_hyperparam_dfs(score, df_min, df_median, df_max, show=True)`
    plots the scores with the output of `get_hyperparam_dfs`
    '''
    hp_xname = df_min.columns.values[0][0]
    hp_yname = df_min.index.values[0][0]
    hp_xvals = [_[1] for _ in df_min.columns.values]
    hp_yvals = [_[1] for _ in df_min.index.values]

    fig, axes = plt.subplots(1, len(df_median.index))
    colors = get_colors(len(hp_yvals))

    for i, yval in enumerate(hp_yvals):
        axs = axes[i]
        flt = df_min.index.get_level_values(1) == yval
        assert flt.any()
        # x = df_median.columns.values.astype(float)
        miny = df_min[flt].values.flatten()
        mediany = df_median[flt].values.flatten()
        maxy = df_max[flt].values.flatten()
        axs.fill_between(hp_xvals, miny, maxy, alpha=0.1, color=colors[i])
        title = "%s=%s" % (hp_yname, str(yval))
        axs.plot(hp_xvals, mediany, linestyle='--', color=colors[i],
                 marker='o', label=title)
        axs.set_title(title.replace('=', ':\n'))
        axs.set_xlabel(hp_xname)
        axs.set_ylim(df_min.values.min(), df_max.values.max())
        axs.grid()
        if i == 0:
            if ylabel:
                axs.set_ylabel(str(ylabel))
        else:
            axs.set_yticklabels([])
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.xlabel('Max samples')
    # plt.ylabel('AUC score')
    # plt.grid()
    plt.tight_layout(rect=[0, 0, 1, 1])
    if show:
        plt.show()
        fig = None

    return fig


def progressbar(length):
    '''`pbar=progressabr(length)`, then into loop: `pbar.update(chunk)`.
    Emulates a progressbar for Python jupyter notebook only
    (`click.progressbar` does not work)
    '''
    class pbar:
        def __init__(self, length):
            self.length = length
            self.progress = 0
            self.time = time.time()
            self.bar_length = 30

        def update(self, value):
            self.progress += value
            val = min(1, np.true_divide(self.progress, self.length))
            block = int(round(self.bar_length * val))
            clear_output(wait=True)
            bar_text = '#' * block + '-' * (self.bar_length - block)
            eta = (self.length - self.progress) * (time.time() - self.time) / self.progress
            eta = timedelta(seconds=int(round(eta)))
            text = "[{0}] {1:.1f}% {2}".format(bar_text, val * 100, str(eta))
            print(text)

    return pbar(length)


def get_pred_dfs(eval_df, show_progress=True):
    '''`get_pred_dfs(eval_df, show_progress=True)` reads and returns a dict of
    file paths mapped to the corresponding Prediction dataframe, for each
    evaluation (row) of `eval_df`. See also `get_eval_df`
    '''
    pbar = progressbar(len(eval_df)) if show_progress else None
    pred_dfs = {}
    for eval_namedtuple in eval_df.itertuples(index=False, name='Evaluation'):
        filepath = _abspath(eval_namedtuple.file_relative_path)
        pred_df = pd.read_hdf(filepath, columns=['outlier', 'predicted_anomaly_score'])
        # filepath2 = join(dirname(filepath), 'allset_test.hdf')  # <- validation set (name misleading)
        # pred_df2 = pd.read_hdf(filepath2, columns=['outlier', 'predicted_anomaly_score'])
        # pred_dfs[filepath] = pd.concat([pred_df, pred_df2], axis=0, sort=False, ignore_index=True, copy=True)
        pred_dfs[eval_namedtuple] = pred_df
        if show_progress:
            pbar.update(1)
    return pred_dfs


def get_eval_df(pred_dfs, evalmetrics=None, show_progress=True):
    '''`get_eval_df(pred_dfs, ems=None)` returns an Evaluation dataframe
    computed on all the Prediction dataframes `pred_dfs`. See also
    `get_pred_dfs`. Once the original `eval_df` has been retireved,
    (`read_summary_eval_df()`) a user can go back and forth in a Notebook to
    retrieve the prediction dataframes (`get_pred_dfs`) and compute new
    evaluation metrics on them with this function (see `EVALMETRICS`).
    '''
    if evalmetrics is None:
        evalmetrics = [EVALMETRICS.AUC, EVALMETRICS.APS, EVALMETRICS.F1MAX]
    else:
        standard_ems = set(_ for _ in EVALMETRICS)
        # keep enum if it is an enum or getit by value if somebody passed a str
        evalmetrics = [_ if _ in standard_ems else EVALMETRICS(_)
                       for _ in evalmetrics]

    data = []
    pbar = progressbar(len(pred_dfs)) if show_progress else None
    for eval_named_tuple, pred_df in pred_dfs.items():
        # convert eval_named_tuple to dct:
        dic = eval_named_tuple._asdict()
        for evalmetric in evalmetrics:
            dic[str(evalmetric)] = evalmetric.compute(pred_df)

        if show_progress:
            pbar.update(1)
#         dic['Area_under_ROC'] = metrics.roc_auc_score(ytrue, ypred)
#         dic['Average_precision_sore'] = metrics.average_precision_score(ytrue,
#                                                                         ypred)
#         pre, rec, thr = metrics.precision_recall_curve(ytrue, ypred)
#         fscores = f1scores(pre, rec)
#         argmax = np.argmax(fscores)
#         dic['F1score_max'] = fscores[argmax]
        data.append(dic)
    dfr = pd.DataFrame(data=data)
    return dfr
#     for col in dfr.columns:
#         printhtml('Models ranked under %s' % col)
#         display(dfr.sort_values([col], ascending=False))
#     
#     printhtml('<h3>Grouping by random_state, computing harmonic mean and printing rankings</h3>')
#     newdata, newindices = [], []
#     for _, df in dfr.groupby(lambda index_value: index_value[:-1]):
#         dic = {c: hmean(df[c].values) for c in df.columns}
#         newdata.append(dic)
#         newindices.append(_)
#     newdfr = pd.DataFrame(index=newindices, data=newdata)
#     for col in newdfr.columns:
#         printhtml('Models ranked under %s' % col)
#         display(newdfr.sort_values([col], ascending=False))


# def filepath2title(filepath):
#     '''`filepath2title(fpath)` converts the `filepath` of a
#     prediction dataframe () into a tuple of hyperparameters
#     `(feats, n_estimators, max_samples, random_state)` (all strings)
#     '''
#     fpathb = os.path.basename(os.path.dirname(filepath))
#     ret = []
#     for key in ['feats'] + list(str(_) for _ in HYPERPARAMS):
#         ret.append(re.search(r'%s=([^&]+)' % key, fpathb).group(1))
#     return tuple(ret)


def plot_freq_distribution(pred_dfs, ncols=None, title_keys=None,
                           mp_hist_kwargs=None, show=True):
    '''`plot_freq_distribution(pred_dfs, ncols=None, mp_hist_kwargs=None, show=True)`
    plots the segments frequency distribution (histogram) for the two classes
    'inliers' and 'outliers'. `pred_dfs` is the output of `get_pred_dfs`.
    '''
    bins = 10
    rows, cols = plotgrid(len(pred_dfs), ncols=ncols)
    fig = plt.figure(constrained_layout=True)
    gsp = fig.add_gridspec(rows, cols)
    idx = 0
    if mp_hist_kwargs is None:
        mp_hist_kwargs = {}
    mp_hist_kwargs.setdefault('density', False)
    mp_hist_kwargs.setdefault('log', False)
    mp_hist_kwargs.setdefault('stacked', False)
    mp_hist_kwargs.setdefault('rwidth', .5)
    for idx, (eval_namedtuple, pred_df) in enumerate(pred_dfs.items()):
        __r, __c = int(idx // cols), int(idx % cols)
        axs = fig.add_subplot(gsp[__r, __c])
        axs.hist(
            [pred_df[~pred_df.outlier].predicted_anomaly_score,
             pred_df[pred_df.outlier].predicted_anomaly_score],
            bins=bins, label=['inliers', 'outliers'], **mp_hist_kwargs
        )
        if title_keys is not None:
            title = "\n".join(getattr(eval_namedtuple, _) for _ in title_keys)
            axs.set_title(title)
        if __r < rows - 1:
            axs.set_xticklabels([])
        else:
            axs.set_xlabel('Score')
        if __c != 0:
            axs.set_yticklabels([])
        else:
            axs.set_ylabel('Instances')
        axs.grid()
    samex(fig.axes)
    samey(fig.axes)
    # fig.tight_layout(rect=[0,0,1,1])
    if show:
        plt.show()
        fig = None

    return fig


def plot_pre_rec_fscore(pred_dfs, ncols=None, title_keys=None,
                        mp_plot_kwargs=None, show=True):
    '''`plot_pre_rec_fscore(pred_dfs, ncols=None, mp_plot_kwargs=None, show=True)`
    plots the segments frequency distribution (histogram) for the two classes
    'inliers' and 'outliers'. `pred_dfs` is the output of `get_pred_dfs`.
    '''
    # ======================================
    # print P, R, Fscore positive label vs thresholds
    # ======================================
    rows, cols = plotgrid(len(pred_dfs), ncols=ncols)
    fig = plt.figure(constrained_layout=True)
    gsp = fig.add_gridspec(rows, cols)
    if mp_plot_kwargs is None:
        mp_plot_kwargs = {}
    mp_plot_kwargs.setdefault(color='red')
    idx = 0
    for idx, (eval_namedtuple, pred_df) in enumerate(pred_dfs.items()):
        __r, __c = int(idx // cols), int(idx % cols)
        axs = fig.add_subplot(gsp[__r, __c])
        prec, rec, thresholds = \
            metrics.precision_recall_curve(pred_df.outlier,
                                           pred_df.predicted_anomaly_score)
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
        prec, rec = prec[:-1], rec[:-1]
        axs.plot(thresholds, prec, label='P[1]', linestyle='--',
                 **mp_plot_kwargs)
        axs.plot(thresholds, rec, label='R[1]', linestyle=':',
                 **mp_plot_kwargs)
        fscores = f1scores(prec, rec)
        axs.plot(thresholds, fscores, label='F[1]', linestyle='-',
                 **mp_plot_kwargs)
        argmax = np.argmax(fscores)
        title = ''
        if title_keys is not None:
            title = "\n".join(getattr(eval_namedtuple, _) for _ in title_keys)
            if title:
                title += '\n'
        title += 'T (best_th) = %.2f' % thresholds[argmax]
        title += '\nP(T) = %.3f' % prec[argmax]
        title += '\nR(T) = %.3f' % rec[argmax]
        title += '\nF(T) = %.3f' % fscores[argmax]
        # title = title.replace('psd@', '').replace('sec', '').replace('\n', ' ')
        axs.set_title(title)
        if __r < rows - 1:
            axs.set_xticklabels([])
        else:
            axs.set_xlabel('Threshold')
        if __c != 0:
            axs.set_yticklabels([])
        else:
            axs.set_ylabel('Value')
        axs.grid()
        axs.legend()
    samex(fig.axes)
    samey(fig.axes)

    if show:
        plt.show()
        fig = None

    return fig


def f1scores(pre, rec):
    '''`f1scores(pre, rec)` returns a numpy array of the f1scores calculated
    element-wise for each precision and recall'''
    f1_ = np.zeros(len(pre), dtype=float)
    isfinite = (pre != 0) & (rec != 0)
    f1_[isfinite] = hmean(np.array([pre[isfinite], rec[isfinite]]), axis=0)
    return f1_


def plotgrid(numaxes, ncols=None):
    '''`plotgrid(numaxes, ncols=None)` returns the tuple (rows, cols) of
    integers denoting the optimal grid to display the given axes in a plot'''
    # get best row/col grid:
    _ = np.ceil(np.sqrt(numaxes))
    assert _ ** 2 >= numaxes

    if ncols is None:
        # fix rows and cols as the given _ number
        rows, cols = int(_), int(_)
        # for some cases (e.g. len(pred_dfs2)==12) we have rows = cols = 4, whereas rows could be 3
        # Do a simple stupid check to see if we can decrement rows:
        while True:
            if (cols - 1) * rows >= numaxes:
                cols -= 1
            else:
                break
    else:
        cols = int(ncols)
        rows = int(np.ceil(np.true_divide(numaxes, cols)))
    return rows, cols


def printdoc():
    '''Prints this table as HTML formatted text (to be used in a Notebook)'''

    # print this doc. Skip all things outside wrapping '---'
    thisdoc = "\n\n" + __doc__[__doc__.find('---') + 3:__doc__.rfind('---')]
    thisdoc += ('\n\nGiven the above definitions, `%s` imported the '
                'following functions/classes/modules '
                'that you can use in this Notebook:') % __name__
    thisdoc = re.sub(r'\*\*([^\*]+)\*\*', r'<b>\1</b>', thisdoc)
    thisdoc = re.sub(r'\*([^\*]+)\*', r'<i>\1</i>', thisdoc)
    thisdoc = re.sub('`([^`]+)`', r'<code>\1</code>', thisdoc)
    thisdoc = re.sub('\n\n+', r'<p>', thisdoc)

    __ret = ("<div style='width:100%;border:1px solid #ddd;overflow:auto;"
             "max-height:40rem'>" + thisdoc + "<table>")
#     __ret = ("<div style='width:100%;border:1px solid #ddd;overflow:auto;"
#              "max-height:30rem'>" + thisdoc + "<table><tr><th>"
#              "Imported function/module/class</th><th>description</th></tr>")
    # re_pattern = re.compile(r'^(.*?)(?:\.\s|\n)')
    for pyobjname, pyobj in globals().items():
        if pyobjname[:1] != '_':
            pyobjname = pyobjname.replace('<', '&lt;').replace('>', '&gt;')
            doc = str(pyobj)
            # if pyobj is not a variable, then print its doc
            ismod, ismeth, isclass, isfunc = \
                (inspect.ismodule(pyobj), inspect.isclass(pyobj),
                 inspect.ismethod(pyobj), inspect.isfunction(pyobj))
            if (ismod or isclass or ismeth or isfunc):
                doc = inspect.getdoc(pyobj) or ''
                # if it's a module or it's imported (not defined here), just
                # show the first part of the doc (up to dot+space or newline):
                if ismod or pyobj.__module__ != __name__:
                    mtch = re.search(r'^(.*?)(?:\.\s|\n)', doc)
                    doc = mtch.group(1) if mtch else doc
                elif not ismod:
                    # is something implemented here, not a simple variable
                    # (method, class, function). wrap `...` in <code> tags:
                    doc = re.sub('`([^`]+)`', r'<code>\1</code>', doc)
            __ret += "<tr><td>%s</td><td>%s</td></tr>" % (pyobjname, doc)
    __ret += '</table></div>'
    printhtml(__ret)


# if __name__ == "__main__":
#     fig, axs = plt.subplot(1)
#     plt.close()
#     fig.show()
    

# printdoc()

# EVALPATH HIDDEN
# REMOVE show from plots
