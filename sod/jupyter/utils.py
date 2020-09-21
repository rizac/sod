'''
`utils.py`: utilities for my notebooks. This module should be used only from
within a Jupyter Notebook.

---
When using the module `share` in a Jupyter Notebook, note the following naming
conventions:

`eval_df` denotes an Evaluation Data frame, i.e. a
tabular object where *a row is a model evaluation* and columns report
evaluation info, e.g., model name, model hyperparams, testset file path,
some evaluation metrics scores.

`pred_df` denotes a Prediction Data frame representing a model evaluation
in details: *a row is a testset instance* and columns
report several instance info, including the instance actual class as boolean
(column 'outlier') and the predicted class/score as float in [0, 1] (column
'predicted_anomaly_score')
---

Both dataframes are the output of several experiments run via
the script `evaluate.py`

Created on 18 Mar 2020

@author: rizac(at)gfz-potsdam.de
'''
from os.path import (join, abspath, dirname, isfile, isdir, basename, dirname,
                     splitext, expanduser, isabs)
import sys
import os
import re
import time
import inspect
from datetime import datetime, timedelta
import contextlib
from enum import Enum
from collections import defaultdict, namedtuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import hmean
from sklearn import metrics
from joblib import dump, load
from IPython.display import display, display_html, clear_output  # https://stackoverflow.com/a/36313217
import pandas as pd
from itertools import cycle
import importlib
from matplotlib.ticker import MultipleLocator


from stream2segment.process.db import Segment, Station, get_session as g_s
import yaml

# for printing, we can do this:
# with pd.option_context('display.max_rows', -1, 'display.max_columns', 5):
# or we simply set once here the max_col_width
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.max_columns', 500)


@contextlib.contextmanager
def dbsession(dbname_or_dbid):
    '''`with dbsession(dbname_or_dbid) as session:`'''
    dbname = dbname_or_dbid
    if not isinstance(dbname_or_dbid, str):
        dbname = [
            'dbpath_eu_new',
            'dbpath_me',
            'dbpath_chile',
            'dbpath_sod_dist_2_20',
            'dbpath_sod_mag_4_5',
            'dbpath_test_dist_around_2000'
        ][dbname_or_dbid-1]

    with open(join(dirname(__file__), 'jnconfig.yaml')) as opn:
        dbases = yaml.safe_load(opn)
    sess = g_s(dbases[dbname], connect_args={'connect_timeout': 10})
    try:
        yield sess
    finally:
        sess.close()
        sess.bind.engine.dispose()


def printhtml(what):
    '''Same as display_html(text, True): prints the html formatted text
    passed as argument'''
    display_html(what, raw=True)


class EVALMETRICS(Enum):
    '''Evaluation metrics enumeration. Each enum item is mapped to a string:
    `AUC` = 'roc_auc_score'
    `APS` = 'average_precision_score'
    `LOGLOSS` = 'log_loss'
    `F1MAX` = 'f1_max' (F1score at `BEST_TH_PR`)
    `PMAX` = 'p_max' (Precision at `BEST_TH_PR`)
    `RMAX` = 'r_max' (Recall at `BEST_TH_PR`)
    `BEST_TH_PR`
     and has a `compute` method
    (e.g., `EVALMETRICS.APS.compute(pred_df)`) which
    returns the metric scalar value from a prediction dataframe
    '''
    # REMEMBER that EVALMETRICS(string) == EVALMETRICS(EVALMETRICS(string))
    APS = 'average_precision_score'
    F1MAX = 'f1_max'
    BEST_TH_PR = 'best_th_pr_curve'
    AUC = 'roc_auc_score'
    LOGLOSS = 'log_loss'
    PMAX = 'p_max'
    RMAX = 'r_max'
    # BEST_TH_ROC = 'best_th_roc_curve'

    def __str__(self):
        return self.value

    def compute(self, pred_df):
        '''Computes the value of this metric'''
        return EVALMETRICS.computeall(pred_df, self)[0]
#         y_true, y_pred = pred_df.outlier, pred_df.predicted_anomaly_score
#         if self == EVALMETRICS.AUC:
#             return metrics.roc_auc_score(y_true, y_pred)
#         if self == EVALMETRICS.APS:
#             return metrics.average_precision_score(y_true, y_pred)
#         if self in (EVALMETRICS.F1MAX, EVALMETRICS.BEST_TH_PR):
#             pre, rec, thr = metrics.precision_recall_curve(y_true, y_pred)
#             fscores = f1scores(pre, rec)
#             argmax = np.argmax(fscores)
#             if self == EVALMETRICS.BEST_TH_PR:
#                 # From the doc: "the last precision and recall values are 1.
#                 # and 0. respectively, and do not have a corresponding
#                 # threshold". Thus if argmax is = len(fscores)-1 we have a
#                 # problem, but this can never happen because f1scores[-1] = 0
#                 return thr[argmax]
#             return fscores[argmax]
# #         if self == EVALMETRICS.BEST_TH_ROC:
# #             fpr, tpr, thr = metrics.roc_curve(y_true, y_pred)
# #             # compute the harmonic mean element-wise, i.e. f1scores(tpr, tnr):
# #             fscores = f1scores(tpr, 1-fpr)
# #             # from the doc: "thr[0] represents no instances being predicted
# #             # and is arbitrarily set to max(y_score) + 1.". Thus we do not
# #             # want to return it, which should never happen becuase scores[0]
# #             # will never be the max, as tpr[0] = 0. However, for safety:
# #             argmax = np.argmax(fscores[1:])
# #             return thr[1 + argmax]
#         if self == EVALMETRICS.LOGLOSS:
#             return metrics.log_loss(y_true, y_pred)
#         raise ValueError('something wrong in EVALMETRIS.compute')

    @classmethod
    def computeall(cls, pred_df, *evalmetrics):
        ret = [np.nan for _ in evalmetrics]
        if pred_df.empty:
            return ret
        y_true, y_pred = pred_df.outlier, pred_df.predicted_anomaly_score
        evalmetrics = [cls(_) for _ in evalmetrics]

        # pre computed precition, recall threshold, if needed:
        pre, rec, pr_ths, fscores, f_argmax = None, None, None, None, None
        if set(evalmetrics) & set((cls.BEST_TH_PR, cls.F1MAX, cls.PMAX, cls.RMAX)):
            pre, rec, pr_ths = metrics.precision_recall_curve(y_true, y_pred)
            fscores = f1scores(pre, rec)
            f_argmax = np.argmax(fscores)

        for i, metric in enumerate(evalmetrics):
            if metric == cls.BEST_TH_PR:
                # From the doc: "the last precision and recall values are 1.
                # and 0. respectively, and do not have a corresponding
                # threshold". Thus if argmax is = len(fscores)-1 we have a
                # problem, but this can never happen because
                # f1scores[-1] = 0
                ret[i] = pr_ths[f_argmax]
            elif metric == cls.F1MAX:
                ret[i] = fscores[f_argmax]
            elif metric == cls.PMAX:
                ret[i] = pre[f_argmax]
            elif metric == cls.RMAX:
                ret[i] = rec[f_argmax]
            elif metric == cls.AUC:
                ret[i] = metrics.roc_auc_score(y_true, y_pred)
            elif metric == cls.APS:
                ret[i] = metrics.average_precision_score(y_true, y_pred)
            elif metric == cls.LOGLOSS:
                ret[i] = metrics.log_loss(y_true, y_pred)
            else:
                raise ValueError('Unrecognized metric: "%s"' % str(metric))
        return ret


def _reorder_eval_df_columns(eval_df, copy=True):
    columns1 = [
        'clf', 'feats', 'n_estimators', 'max_samples', 'random_state',
        *[str(_) for _ in EVALMETRICS]
    ]
    columns1 = [_ for _ in columns1 if _ in eval_df.columns]
    columns2 = sorted(_ for _ in eval_df.columns if _ not in columns1)
    ret = eval_df[columns1 + columns2]
    return ret if not copy else ret.copy()


def read_summary_eval_df(**kwargs):
    '''`read_summary_eval_df(**kwargs)` = pandas
    `read_hdf('../evaluations/results/summary_evaluationmetrics.hdf', **kwargs)`
    reads and returns the Evaluation dataframe created and incremented by each
    execution of the main script `evaluate.py`. *Thus, if no
    evaluation has been run on this computer, no evaluation dataframe exists
    and the Notebook using this module will not work.*
    Keyword arguments of the functions are the same keyword arguments as pandas
    `read_hdf`.
    '''
    return read_eval_df('summary_evaluationmetrics.hdf', **kwargs)
    # dfr = pd.read_hdf(_abspath('summary_evaluationmetrics.hdf'), **kwargs)
    # # _key is the prediction dataframe path, relative to
    # # the EVALPATH directory (see _abspath). Create a new filepath column with
    # # the complete path of each prediction:
    # # fAlso, consider renaming leading underscores from pandas columns
    # # otherwise itertuples does not work (namedtuples prohibit leading
    # # underscores in attributes)
    # dfr.rename(columns={'_key': 'relative_filepath'}, inplace=True)
    # return _reorder_eval_df_columns(dfr)


def read_eval_df(path_or_name, **kwargs):
    if not isabs(path_or_name):
        path_or_name = _abspath(path_or_name)
    dfr = pd.read_hdf(path_or_name, **kwargs)
    # _key is the prediction dataframe path, relative to
    # the EVALPATH directory (see _abspath). Create a new filepath column with
    # the complete path of each prediction:
    # fAlso, consider renaming leading underscores from pandas columns
    # otherwise itertuples does not work (namedtuples prohibit leading
    # underscores in attributes)
    dfr.rename(columns={'_key': 'relative_filepath'}, inplace=True)
    return _reorder_eval_df_columns(dfr)


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


def load_clf(relativepath):
    '''`load_clf(relativepath)` loads a classifier from a path relative to
    the evaluation directory of this package (where `evaluate.py` saved
    evaluation results)
    '''
    return load(_abspath(relativepath))


def read_pred_df(relativepath, **kwargs):
    '''`red_pred_df(relativepath, **kwargs)` =
    pandas `read_hdf('../evaluations/results' + relativepath, **kwargs)`
    reads and returns a Prediction dataframe created via the main script
    `evaluate.py`.
    The keyword argument `columns`, if not specified, defaults to the minimal
    `('outlier', 'predicted_anomaly_score')`. Pass None to load all columns, or
    a list of columns as in pandas documentation
    '''
    kwargs.setdefault('columns', ('outlier', 'predicted_anomaly_score'))
    return pd.read_hdf(_abspath(relativepath), **kwargs)


def samex(axes):
    '''`samex(axes)` sets the same x limits on all matplotlib Axes provided
    as argument'''
    return _sameaxis(axes, 'x')


def samey(axes):
    '''`samey(axes)` sets the same x limits on all matplotlib Axes provided
    as argument'''
    return _sameaxis(axes, 'y')


def _sameaxis(axes, meth):
    lmin, lmax = None, None
    if axes is None or not len(axes):  # pylint: disable=len-as-condition
        return lmin, lmax
    lims = np.array([_.get_xlim() for _ in axes]) if meth == 'x' else \
        np.array([_.get_ylim() for _ in axes])
    minlims, maxlims = lims[:, 0], lims[:, 1]
    if not np.isnan(minlims).all() and not np.isnan(maxlims).all():
        lmin, lmax = np.nanmin(minlims), np.nanmax(maxlims)
        if lmax > lmin:
            for ax_ in axes:
                if meth == 'x':
                    ax_.set_xlim(lmin, lmax)
                else:
                    ax_.set_ylim(lmin, lmax)
    return lmin, lmax


def _get_fig_and_axes(show_argument, default_rows, default_columns,
                      grid_direction_horizontal=True):
    # Used only if default_rows  and default_columns !=1
    fig = None
    if show_argument in (True, False):
        fig = plt.figure(constrained_layout=True)
        gsp = fig.add_gridspec(default_rows, default_columns)
        if default_rows != 1 and default_columns != 1:
            if grid_direction_horizontal:
                axes = [fig.add_subplot(gsp[r, c])
                        for r in range(default_rows)
                        for c in range(default_columns)]
            else:
                axes = [fig.add_subplot(gsp[r, c])
                        for c in range(default_columns)
                        for r in range(default_rows)]
        elif default_columns == 1:
            axes = [fig.add_subplot(gsp[r, 0]) for r in range(default_rows)]
        else:
            axes = [fig.add_subplot(gsp[0, c]) for c in range(default_columns)]
    else:
        axes_class = matplotlib.axes.Axes  # @UndefinedVariable
        if not hasattr(show_argument, '__len__') or \
                not all(isinstance(_, axes_class) for _ in show_argument):
            raise ValueError('`show` argument must be True, False, '
                             'or a list of matplotlib Axes')
        elif len(show_argument) != default_rows * default_columns:
            raise ValueError('`show` argument number of Axes (%d) '
                             'should be %d' % (len(axes), default_rows *
                                               default_columns))
        axes = show_argument

    return fig, axes


@contextlib.contextmanager
def rcparams(rcparams=None, figsizeratio=(1, 1)):
    '''`with rcparams(rcparams={}, figsizeratio=(1.0, 1.0)):`
    makes temporarily matplotlib rcParams.
    Make sure to run this after %matplotlib inline.
    For info see https://stackoverflow.com/questions/36367986/how-to-make-inline-plots-in-jupyter-notebook-larger'''
    rcparams = rcparams or {}
    defparams = {k: plt.rcParams[k] for k in rcparams}
    figsize = 'figure.figsize'
    wh = plt.rcParams[figsize]
    defparams[figsize] = tuple(wh)
    try:
        figsizeratio = tuple(figsizeratio)
    except TypeError:
        figsizeratio = (figsizeratio, figsizeratio)
    figsizeratio = tuple(1 if _ is None else float(_) for _ in figsizeratio)

    if figsizeratio != (1, 1):
        w, h = wh[0], wh[1]
        area = w * h
        relw, relh = figsizeratio
        if relw is not None and relh is not None:
            w *= relw
            h *= relh
        elif relw is not None:
            w *= relw
            h = area / w
        elif relh is not None:
            h *= relh
            w = area / h
        rcparams[figsize] = (w, h)
    for k, v in rcparams.items():
        plt.rcParams[k] = v
    try:
        yield
    finally:
        for k, v in defparams.items():
            plt.rcParams[k] = v


def plot_feats_vs_evalmetrics(eval_df, evalmetrics=None,
                              ylabel=lambda metric: str(metric).replace('_', ' '),
                              mpl_rect_args=None,
                              mpl_plot_args=None,
                              show=True):
    '''`plot_feats_vs_evalmetrics(eval_df, evalmetrics=None, ylabel=lambda metric:..., mpl_rect_args={}, mpl_plot_args={}, show=True)` plots the
    features of `eval_df` grouped and
    colored by PSD period counts versus the given `ems` (Evaluation metrics,
    passes as strings or `EVALMETRICS` enum items. None - the default -
    shows all possile metrics stored in a `eval_df`: Area under ROC curve,
    Average precision score, log loss)
    '''
    # convert to EVALMETRICS instances:
    if evalmetrics is None:
        evalmetrics = [EVALMETRICS.APS, EVALMETRICS.AUC]
    else:
        evalmetrics = [EVALMETRICS(_) for _ in evalmetrics]
    feats = _unique_sorted_features(eval_df)
    feat_labels = [
        _.replace('psd@', '').replace('sec', '').replace(',', '  ')
        for _ in feats
    ]

    colors = _get_colors(max(len(_.split(',')) for _ in feats), .4, .85)
#     fig = plt.figure(constrained_layout=True)
#     gsp = fig.add_gridspec(len(evalmetrics), 1)

    fig, axes = _get_fig_and_axes(show, len(evalmetrics), 1,
                                  grid_direction_horizontal=False)

    mpl_rect_args = mpl_rect_args or {}
    for k, v in dict(width=.8, fill=True, facecolor='white',  # linewidth=1.5,
                     zorder=10).items():
        mpl_rect_args.setdefault(k, v)

    mpl_plot_args = mpl_plot_args or {}
    for k, v in dict(marker='.', markersize=9,
                     linewidth=0, mew=2, zorder=20).items():
        mpl_plot_args.setdefault(k, v)

    for j, metric in enumerate(evalmetrics):
        metric_name = str(metric)
        axs = axes[j]

        for i, feat in enumerate(feats):
            df_ = eval_df[eval_df.feats == feat][metric_name]
            min_, median, max_ = df_.min(), df_.median(), df_.max()

            xerr = [[median-min_], [max_-median]]
            color = colors[len(feat.split(',')) - 1]

            #  NOTES ON THE ARGUMENTS OF errorbar BELOW:
            #
            #  eline ->   |-----------o----------|
            #
            #             ^           ^
            #            cap       marker


#             axs.errorbar(
#                 median,
#                 i,
#                 xerr=xerr,
#                 color=color,  # tuple(list(color)[:-1] + [0.6]),
#                 elinewidth=15,  # <- eline vertical height
#                 marker='|',
#                 markersize=15,  # <- size (if marker= "|', -> vertical height)
#                 mec=[0, 0, 0],
#                 # mfc= [0, 0, 0],
#                 mew=2,  # <- marker horiz. width IT OVERRIDES capthick!!
#                 capthick=0,  # <- cap horizontal width
#                 capsize=0,  # <- cap vertical height
#             )

            # instead of the errorbars above, quite ugly, we display rectangles
            # and a bar for the median:
            rect = matplotlib.patches.Rectangle([i-0.4, min_],
                                                height=max_-min_,
                                                edgecolor=color,
                                                **mpl_rect_args)

            axs.add_patch(rect)

            axs.plot([i], [median], color=color, **mpl_plot_args)

        axs.set_xticks(list(range(len(feats))))
        if j == len(evalmetrics) - 1:
            axs.set_xlabel('Features (PSD periods)')
            axs.set_xticklabels(feat_labels, rotation=80)
        else:
            axs.set_xticklabels([])

        axs.set_ylabel(ylabel(metric))
        axs.grid(zorder=0)

    if show is True:
        plt.show()
        fig = None

    return fig  # it is not None only if show=False


_DEFAULT_COLORMAP = None  # the default is ''cubehelix', see _get_colors


# @contextlib.contextmanager
# def _use_tmp_colormap(name):
#     '''`with use_tmp_colormap(name)` changes temporarily the colormap. Useful
#     before plotting
#     '''
#     global _DEFAULT_COLORMAP  # pylint: disable=global-statement
#     _ = _DEFAULT_COLORMAP
#     _DEFAULT_COLORMAP = name
#     try:
#         yield
#     finally:
#         _DEFAULT_COLORMAP = _


def _get_colors(numcolors, min=None, max=None):
    '''`get_colors(N)` returns N different colors for plotting
    `get_colors(N, min, max)` does the same but colors are visible also in
    black and white. `min` in [0, 1] controls how dark is the darkest color,
    `max` in [0, 1] controls how "light" is the brightest color
    '''
    if _DEFAULT_COLORMAP is None and min is None and max is None:
        # use tab10 as default colormap, returning 10 colors
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i, 1) for i in np.arange(0, 1.1, 0.1)]
        ret = []
        for i, c in enumerate(cycle(colors), 1):
            color = list(matplotlib.colors.rgb_to_hsv(c[:3]))
            color[1] -= color[1]*.5
            ret.append(matplotlib.colors.hsv_to_rgb(color[:3]))
            if i >= numcolors:
                break
        return ret

    # provide the default colormap 'cubehelix'
    cmap = plt.get_cmap(_DEFAULT_COLORMAP or 'cubehelix')
    # with _DEFAULT_COLORMAP,
    # colors above 0.9 are too light, thus we shrink the range
    return [cmap(i, 1)
            for i in np.linspace(min, max, numcolors, endpoint=True)]


# make getcolors public available:
get_colors = _get_colors


def _unique_sorted_features(eval_df):
    '''`_unique_sorted_features(eval_df)` returns a list of sorted strings
    representing the unique features of the DataFrame passed as argument.
    Features are sorted by number of PSD periods and if equal, by periods sum
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
    # convert to EVALMETRICS instances:
    evalmetric = EVALMETRICS(evalmetric)

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


def plot_hyperparam_dfs(df_min, df_median, df_max, ylabel=None,
                        mpl_plot_args=None, show=True):
    '''`plot_hyperparam_dfs(df_min, df_median, df_max, ylabel=None, mpl_plot_args={}, show=True)`
    plots the scores with the output of `get_hyperparam_dfs` producing
    `N` plots where for i=1 to N, the i-th plot displays the i-th row of
    dfmedian, dfmin, and dfmax are plotted (dfmin and dfmax as shaded area,
    dfmedian as scatter plot with lines on top)
    '''
    hp_xname = df_min.columns.values[0][0]
    hp_yname = df_min.index.values[0][0]
    hp_xvals = [_[1] for _ in df_min.columns.values]
    hp_yvals = [_[1] for _ in df_min.index.values]

    mpl_plot_args = mpl_plot_args or {}
    for k, v in dict(linestyle='--', marker='o').items():  #, linewidth=2):
        mpl_plot_args.setdefault(k, v)

#     fig, axes = plt.subplots(1, len(df_median.index))
    fig, axes = _get_fig_and_axes(show, 1, len(df_median.index),
                                  grid_direction_horizontal=True)
    colors = _get_colors(len(hp_yvals))

    for i, yval in enumerate(hp_yvals):
        axs = axes[i]
        flt = df_min.index.get_level_values(1) == yval
        assert flt.any()
        # x = df_median.columns.values.astype(float)
        miny = df_min[flt].values.flatten()
        mediany = df_median[flt].values.flatten()
        maxy = df_max[flt].values.flatten()
        axs.fill_between(hp_xvals, miny, maxy, alpha=0.2, color=colors[i])
        title = "%s=%s" % (hp_yname, str(yval))
        axs.plot(hp_xvals, mediany, color=colors[i], label=title,
                 **mpl_plot_args)
        axs.set_title(title)  # .replace('=', ':\n'))
        axs.set_xlabel(hp_xname)
        axs.set_ylim(df_min.values.min(), df_max.values.max())
        axs.grid()
        if i == 0:
            if ylabel:
                axs.set_ylabel(str(ylabel))
        else:
            axs.set_yticklabels([])

    # plt.tight_layout(rect=[0, 0, 1, 1])
    if show is True:
        plt.show()
        fig = None

    return fig


def progressbar(length):
    '''`pbar=progressabr(length)`, then into loop: `pbar.update(chunk)`.
    Emulates a progressbar for Python jupyter notebook only
    (`click.progressbar` does not work)
    '''

    class t:
        '''this class is a wrapper around a string redefining the __repr__
        without leading and trailing single quotes ' so that display below
        does not shows them
        '''
        def __init__(self, string):
            self.string = string

        def __repr__(self):
            return self.string

    class pbar:
        def __init__(self, length):
            self.length = length
            self.progress = 0
            self.starttime = time.time()
            self.bar_length = 30
            self._d_id = None
            self._chfill = "●"  # '#'
            self._chempty = "○"  # '-'

        def update(self, value):
            self.progress += value
            val = min(1, np.true_divide(self.progress, self.length))
            elapsed_t = time.time() - self.starttime
            eta = (self.length - self.progress) * elapsed_t / self.progress
            eta = timedelta(seconds=int(round(eta)))

            block_fill = int(round(self.bar_length * val))
            block_empty = self.bar_length - block_fill
            bar_ = self._chfill * block_fill + self._chempty * block_empty
            text = "[{0}] {1:.1f}% {2}".format(bar_, val * 100, str(eta))

            # We previously used clear_output(wait=True) before displaying
            # the progressbar. It works but it clears the whole notebook cell,
            # including text written before the bar shows up.
            # Following https://stackoverflow.com/a/57096808
            # we have to get the id of the displayed text update it, which
            # clears and repaints onlt the progressbar. Note that we tried with
            # display_html but it does not work
            if self._d_id is None:
                self._d_id = display(t(text), display_id=True)
            else:
                self._d_id.update(t(text))

    return pbar(length)


def get_pred_dfs(eval_df, postfunc=None, show_progress=True, **kwargs):
    '''`get_pred_dfs(eval_df, postfunc=None, show_progress=True)` reads and
    returns a dict of file paths mapped to the corresponding Prediction
    dataframe, for each evaluation (row) of `eval_df`. See also `get_eval_df`.
    `postfunc`, if supplied, is a `func(keytuple, pred_df)` called on each
    prediction dataframe read via `read_pred_df(keytuple.relative_filepath)`.
    `keytuple` is a namedtuple representing the `eval_df` fields of `pred_df`
    '''
    pbar = progressbar(len(eval_df)) if show_progress else None
    pred_dfs = {}
    for eval_namedtuple in eval_df.itertuples(index=False, name='Evaluation'):
        pred_df = read_pred_df(eval_namedtuple.relative_filepath, **kwargs)
        if postfunc is not None:
            pred_df = postfunc(eval_namedtuple, pred_df)

        pred_dfs[eval_namedtuple] = pred_df
        if show_progress:
            pbar.update(1)
    return pred_dfs


def get_eval_df(pred_dfs, evalmetrics=None, show_progress=True):
    '''`get_eval_df(pred_dfs, evalmetrics=None, show_progress=True)` returns
    an Evaluation dataframe
    computed on all the Prediction dataframes `pred_dfs`. See also
    `get_pred_dfs`. Once the original `eval_df` has been retireved,
    (`read_summary_eval_df()`) a user can go back and forth in a Notebook to
    retrieve the prediction dataframes (`get_pred_dfs`) and compute new
    evaluation metrics on them with this function (see `EVALMETRICS`).
    '''
    # convert to EVALMETRICS instances:
    if evalmetrics is None:
        evalmetrics = [EVALMETRICS.APS, EVALMETRICS.F1MAX, EVALMETRICS.AUC]
    else:
        evalmetrics = [EVALMETRICS(_) for _ in evalmetrics]

    data = []
    pbar = progressbar(len(pred_dfs)) if show_progress else None
    for eval_named_tuple, pred_df in pred_dfs.items():
        # convert eval_named_tuple to dct:
        dic = eval_named_tuple._asdict()

        # remove old eval metrics:
        for _ in EVALMETRICS:
            dic.pop(str(_), None)

        # compute new ones:
        metricsvalues = EVALMETRICS.computeall(pred_df, *evalmetrics)
        for evalmetric, value in zip(evalmetrics, metricsvalues):
            dic[str(evalmetric)] = value
#         for evalmetric in evalmetrics:
#             dic[str(evalmetric)] = evalmetric.compute(pred_df)

        if show_progress:
            pbar.update(1)

        data.append(dic)
    dfr = pd.DataFrame(data=data)
    return _reorder_eval_df_columns(dfr)


def _rank_eval(eval_df, evalmetrics, columns=None, mean='hmean'):
    '''`rank_eval(eval_df, evalmetrics, columns=None, mean='hmean')` returns
    a dict with each metric in `metrics` mapped to an Evaluation dataframe
    with the metric scores sorted descending. `columns` is optional and used
    to group rows of `eval_df` first merging them into a single-row dataframe
    with the metric column reporting the computed `mean` ('hmean' for harmonic
    mean, 'mean' for arithmetic mean, or 'median') on that group.
    '''
    # convert to EVALMETRICS instances:
    evalmetrics = [EVALMETRICS(_) for _ in evalmetrics]

    if columns is not None:
        metric2df = {str(m): [] for m in evalmetrics}
        for _, dfr in eval_df.groupby(columns):
            for metric in metric2df.keys():
                values = dfr[metric]
                finite = pd.notna(values)
                meanvalue = 0
                if finite.any():
                    if mean == 'hmean':
                        values = values[finite]
                        nonzero = values != 0
                        if nonzero.any():
                            values = values[nonzero].values
                            meanvalue = hmean(values)
                    elif mean == 'mean':
                        meanvalue = values.mean(skipna=True)
                    else:  # median
                        meanvalue = values.median(skipna=True)

                # take the first dataframe and add the metric with
                # meanvalue
                dfr = dfr.iloc[0: 1].copy()
                dfr[metric] = meanvalue
                # append to metrics:
                dfr_columns = columns + [metric]
                rem_columns = [_ for _ in eval_df.columns if _ not in dfr_columns]
                metric2df[metric].append(dfr[dfr_columns + rem_columns])

        for metric in metric2df.keys():
            metric2df[metric] = \
                pd.concat(metric2df[metric],
                          axis=0,
                          sort=False,
                          ignore_index=True)
    else:
        metric2df = {str(m): eval_df for m in evalmetrics}

    for metric in metric2df.keys():
        metric2df[metric] = _reorder_eval_df_columns(
            metric2df[metric].sort_values([metric], ascending=False)
        )
    return metric2df


#    rows, cols = grid4plot(len(pred_dfs), ncols=ncols)
#    fig = plt.figure(constrained_layout=True)
#    gsp = fig.add_gridspec(rows, cols)
#
# for idx, (key, pred_df) in enumerate(pred_dfs.items()):
#     __r, __c = int(idx // cols), int(idx % cols)
#     axs = fig.add_subplot(gsp[__r, __c])
#
#     ... (plot) ...
#
#     if __r < rows - 1:
#         axs.set_xticklabels([])
#     else:
#         axs.set_xlabel('Score')
#     if __c != 0:
#         axs.set_yticklabels([])
#     else:
#         axs.set_ylabel('Instances')
#     axs.grid()
# samex(fig.axes)
# samey(fig.axes)

def plot_freq_distribution(pred_df, axs, bins=None, title=None,
                           mpl_hist_args=None):
    '''`plot_freq_distribution(pred_dfs, ncols=None, titles=str, mpl_hist_args=None, show=True)`
    plots the segments frequency distribution (histogram) for the two classes
    'inliers' and 'outliers'. `pred_dfs` is a dict of keys mapped to
    a prediction dataframe (see e.g. output of `get_pred_dfs`). `titles` is
    a function that will be called on each key of `pred_dfs` and should return
    a string. If missing, it defaults to `str(key)`
    '''
    mpl_hist_args = mpl_hist_args or {}
    for k, v in dict(density=False, log=False, stacked=False,
                     rwidth=.75).items():
        mpl_hist_args.setdefault(k, v)

    # mp_hist_kwargs.setdefault('linewidth', 2)
    # mp_hist_kwargs.setdefault('color', )

#     if 'color' not in mp_hist_kwargs:
#         colors = _get_colors(2)
#     else:
#         colors = [mp_hist_kwargs['color'], mp_hist_kwargs['color']]

#     axs.hist(
#         pred_df[~pred_df.outlier].predicted_anomaly_score,
#         bins=bins, label='inliers',
#         **{**mp_hist_kwargs, 'edgecolor': colors[0], 'color': 'white'}
#     )
#     axs.hist(
#         pred_df[pred_df.outlier].predicted_anomaly_score,
#         bins=bins, label='outliers',
#         **{**mp_hist_kwargs, 'color': colors[1], 'edgecolor': colors[1]}
#     )

    hist_bins = bins
    if bins is None:
        step = 0.01
        bmin = step * int(np.floor(pred_df.predicted_anomaly_score.min()/step))
        bmax = step * int(np.ceil(pred_df.predicted_anomaly_score.max()/step))
        hist_bins = np.arange(bmin, bmax+step, step)

#     axs.hist(
#         [pred_df[~pred_df.outlier].predicted_anomaly_score,
#          pred_df[pred_df.outlier].predicted_anomaly_score],
#         bins=hist_bins, label=['inliers', 'outliers'],
#         **mp_hist_kwargs
#     )
    axs.hist(
        [pred_df[~pred_df.outlier].predicted_anomaly_score,
         pred_df[pred_df.outlier].predicted_anomaly_score],
        bins=hist_bins, label=['inliers', 'outliers'],
        **mpl_hist_args
    )
    axs.hist(
        [pred_df[~pred_df.outlier].predicted_anomaly_score,
         pred_df[pred_df.outlier].predicted_anomaly_score],
        bins=hist_bins, label=['inliers', 'outliers'],
        **mpl_hist_args
    )

    if title:
        axs.set_title(title)

    axs.ticklabel_format(axis='y', style='sci', scilimits=(-4, 4))

    if bins is None:
        axs.xaxis.set_minor_locator(MultipleLocator(0.05))
        # axs.grid(axis="x", which='minor', linestyle='-')

#     if __r < rows - 1:
#         axs.set_xticklabels([])
#     else:
#         axs.set_xlabel('Score')
#     if __c != 0:
#         axs.set_yticklabels([])
#     else:
#         axs.set_ylabel('Instances')

    axs.set_xlabel('score')
    axs.set_ylabel('num. segments')
    # axs.grid()
    return axs


def argwhere_array_equals_value(array, value):
    '''`argwhere_array_equals_value(array, value)` = numpy array of integers
    finds where array equals value, approximating with linear interpolation
    the zones where array crosses `value`
    '''
    # https://stackoverflow.com/a/48901879

    # Find where sign of array - value is changing:
    r = np.where(np.diff(np.sign(array - value)) != 0)
    # Take the value one index below and above the transition and interpolate.
    idx = r + (value - array[r]) / (array[r + np.ones_like(r)] - array[r])
    # Add to `idx` where the value is actually equals the value:
    idx = np.append(idx, np.where(array == value))
    # sort indices:
    idx = np.sort(idx)
    # return integers (approximation):
    return np.round(idx, 0).astype(int)


from obspy.signal.spectral_estimation import get_nlnm, get_nhnm  # @IgnorePep8


def get_petterson_bounds(periods):
    '''`get_petterson_bounds([p1, ...pN])` -> [lowbound1, ... lowboundN], [highbound1, ..., highboundN]
    `get_petterson_bounds(p)` -> lowbound, highbound
    '''
    l_periods, l_psd = get_nlnm()
    h_periods, h_psd = get_nhnm()
    periodz = np.log10(periods)
    return np.interp(periodz, np.log10(l_periods[::-1]), l_psd[::-1]), \
        np.interp(periodz, np.log10(h_periods[::-1]), h_psd[::-1])


def read_handlabelled_channels():
    '''`read_handlabelled()` reads the annotated segments
    from a dedicated directory and returns a dataframe with
    columns: dataset_id (int), start_time (datetime), end_time (datetime),
    outlier (bool), station_id (int)
    location_code (str), channel_code (str)
    '''
    _root = join(expanduser('~'), 'Nextcloud', 'rizac', 'outliers_paper')
    annotations = [join(_root, _)
                   for _ in ['0.75_0.85_dino.csv',
                             '0.65_0.75_dino.csv',
                             '0.55_0.65_dino.csv']]

    annotations = [pd.read_csv(_) for _ in annotations]
    for i, annot in enumerate(annotations):
        annot['score_d'] = pd.to_numeric(annot['score_d'], errors='coerce')
        annot = annot[pd.notna(annot.score_d)]
        annotations[i] = annot
    annotations = pd.concat(annotations, axis=0, sort=False)

    ret = {
        'outlier': annotations.score_d >= 1,
        'station_id': annotations.station_id.astype(int),
        'location_code': annotations['location.channel'].str.split('.').str[0],
        'channel_code': annotations['location.channel'].str.split('.').str[1],
        'dataset_id': annotations.dataset_id.astype(int),
        'start_time': pd.to_datetime(annotations.start_time),
        'end_time': pd.to_datetime(annotations.end_time)
    }

    ret = pd.DataFrame(ret)
    ret.channel_code = ret.channel_code.astype('category')
    ret.location_code = ret.location_code.astype('category')
    ret.loc[pd.isna(ret.start_time), 'start_time'] = datetime(1900, 1, 1)
    ret.loc[pd.isna(ret.end_time), 'end_time'] = \
        datetime.utcnow() + timedelta(days=365)
    return ret


def read_handlabelled_segments():
    '''`read_handlabelled()` reads the annotated segments
    from a dedicated directory and returns a dataframe with
    columns: dataset_id (int), 'Segment.db.id' (int), 'outlier' (bool, always
    True)
    '''
    # set datetime to not be null:
    _root = join(expanduser('~'), 'Nextcloud', 'rizac', 'outliers_paper')
    single_annots = [join(_root, _) for _ in ['lista_1_out_dino',
                                              'lista_2_out_dino']]
    dataset_ids, seg_ids = [], []
    for dataset_id, annot in enumerate(single_annots, 1):
        with open(annot, 'r') as opn:
            content = opn.read().strip().split(' ')
        segids = [int(_) for _ in content]
        dataset_ids += [dataset_id] * len(segids)
        seg_ids += segids

    outlier = [True] * len(dataset_ids)

    return pd.DataFrame({'Segment.db.id': seg_ids,
                         'dataset_id': dataset_ids,
                         'outlier': outlier})



def heatmap_df(dfr, col_x, col_y, bins_x=10, bins_y=10):

    cols = [col_x, col_y]
    dfr = dfr.dropna(subset=cols)

#     qcuts = [pd.cut(dfr[k], 100, duplicates='drop') for k in cols]
#     # make a series with bins (one for each column) -> num of pts per bins:
#     series_df_bins = dfr.groupby(qcuts).size()
#     # convert to dataframe with columns [col_x, coly] mapped to an interval
#     # hmdf has a column "0" with the counts according to that interval
#     class_df_bins = series_df_bins.reset_index()
#     # drop zeros:
#     class_df_bins = class_df_bins[class_df_bins[0] > 0]
    
    if type(bins_x) == int or isinstance(bins_x, np.integer):
        binsx = pd.interval_range(dfr[col_x].min()-1e-5, dfr[col_x].max(),
                                  periods=bins_x+1, name=col_x)
    else:
        pdarrays = pd.arrays  # @UndefinedVariable
        binsx = pdarrays.IntervalArray.from_arrays(bins_x[:-1], bins_x[1:])

    if type(bins_y) == int or isinstance(bins_y, np.integer):
        binsy = pd.interval_range(dfr[col_y].min()-1e-5, dfr[col_y].max(),
                                  periods=bins_y+1, name=col_y)
    else:
        pdarrays = pd.arrays  # @UndefinedVariable
        binsy = pdarrays.IntervalArray.from_arrays(bins_y[:-1], bins_y[1:])

    ret = pd.DataFrame(index=binsy, columns=binsx, data=0)
    ret.index.name = col_y
    ret.columns.name = col_x

    for binx in binsx:
        flt1 = dfr[col_x] >= binx.left if binx.closed_left else dfr[col_x] > binx.left
        flt2 = dfr[col_x] <= binx.right if binx.closed_right else dfr[col_x] < binx.right
        count0 = (flt1 & flt2).sum()
        for biny in binsy:
            if count0 == 0:
                count = 0
            else:
                flt3 = dfr[col_y] >= biny.left if biny.closed_left else dfr[col_y] > biny.left
                flt4 = dfr[col_y] <= biny.right if biny.closed_right else dfr[col_y] < biny.right
                count = (flt1 & flt2 & flt3 & flt4).sum()
            ret.loc[biny, binx] = count
    return ret
            

    
    
    
    
# def plot_1dclf_scores_vs_psdperiods(clf, axs, xaxis_periods=None, title=None,
#                                     mp_hist_kwargs=None, petterson_period=None,
#                                     petterson_mp_kwargs=None):
#     '''`plot_freq_distribution(pred_dfs, ncols=None, titles=str, mp_hist_kwargs=None, show=True)`
#     plots the segments frequency distribution (histogram) for the two classes
#     'inliers' and 'outliers'. `pred_dfs` is a dict of keys mapped to
#     a prediction dataframe (see e.g. output of `get_pred_dfs`). `titles` is
#     a function that will be called on each key of `pred_dfs` and should return
#     a string. If missing, it defaults to `str(key)`
#     '''
#     if xaxis_periods is None:
#         xaxis_periods = np.arange(-250, 0, 0.1)
# 
#     psdvalues_ = np.asarray(xaxis_periods).reshape(len(xaxis_periods), 1)
# 
#     if mp_hist_kwargs is None:
#         mp_hist_kwargs = {}
# #     mp_hist_kwargs.setdefault('density', False)
# #     mp_hist_kwargs.setdefault('log', False)
# #     mp_hist_kwargs.setdefault('stacked', False)
# #     mp_hist_kwargs.setdefault('rwidth', .75)
#     mp_hist_kwargs.setdefault('linewidth', 2)
#     mp_hist_kwargs.setdefault('zorder', 100)
# 
#     axs.plot(
#         xaxis_periods,
#         -clf.score_samples(psdvalues_),
#         **{**mp_hist_kwargs}
#     )
# 
#     if petterson_period is not None:
# 
#         pbounds = get_petterson_bounds(petterson_period)
#         if not petterson_mp_kwargs:
#             petterson_mp_kwargs = {}
#         petterson_mp_kwargs.setdefault('alpha', 0.25)
#         petterson_mp_kwargs.setdefault('color', 'gray')
# 
#         # petterson_mp_kwargs.setdefault('zorder', 10)
#         axs.axvspan(pbounds[0], pbounds[1], **petterson_mp_kwargs)
# 
# 
# #         axs.hist(
# #             [pred_df[~pred_df.outlier].predicted_anomaly_score,
# #              pred_df[pred_df.outlier].predicted_anomaly_score],
# #             bins=bins, label=['inliers', 'outliers'],
# #             **mp_hist_kwargs
# #         )
# 
#     if title:
#         axs.set_title(title)
# #     if __r < rows - 1:
# #         axs.set_xticklabels([])
# #     else:
# #         axs.set_xlabel('Score')
# #     if __c != 0:
# #         axs.set_yticklabels([])
# #     else:
# #         axs.set_ylabel('Instances')
#     axs.grid()
#     return axs


def plot_pre_rec_fscore(pred_df, axs, title=None,
                        mpl_plot_args=None):
    '''`plot_pre_rec_fscore(pred_dfs, ncols=None, titles=str, mp_plot_kwargs=None, show=True)`
    plots the segments frequency distribution (histogram) for the two classes
    'inliers' and 'outliers'. `pred_dfs` is a dict of keys mapped to
    a prediction dataframe (see e.g. output of `get_pred_dfs`). `titles` is
    a function that will be called on each key of `pred_dfs` and should return
    a string. If missing, it defaults to `str(key)`
    '''
    mpl_plot_args = mpl_plot_args or {}
    # mp_plot_kwargs.setdefault('linewidth', 2)

    prec, rec, thresholds = \
        metrics.precision_recall_curve(pred_df.outlier,
                                       pred_df.predicted_anomaly_score)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
    prec, rec = prec[:-1], rec[:-1]
    fscores = f1scores(prec, rec)
    argmax = np.argmax(fscores)
    bth = thresholds[argmax]

    axs.plot(thresholds, prec, label='P(%.2f)=%.3f' % (bth, prec[argmax]),
             linestyle='--', **mpl_plot_args)
    axs.plot(thresholds, rec, label='R(%.2f)=%.3f' % (bth, rec[argmax]),
             linestyle='-.', **mpl_plot_args)
    axs.plot(thresholds, fscores, label='F(%.2f)=%.3f' % (bth, fscores[argmax]),
             linestyle='-', **mpl_plot_args)

#     if title:
#         argmax = np.argmax(fscores)
#         title += '\nat threshold %.2f:' % thresholds[argmax]
#         title += '\nP=%.3f' % prec[argmax]
#         title += ' R=%.3f' % rec[argmax]
#         title += ' F=%.3f' % fscores[argmax]
    if title:
        axs.set_title(title)

    step = 0.1
    #
    # xmin = step * int(np.floor(thresholds[0]/step))
    # xmax = step * int(np.ceil(thresholds[-1]/step))
    axs.xaxis.set_minor_locator(MultipleLocator(step))
    axs.set_xticks([0.5, np.round(bth, 2)])

    axs.yaxis.set_minor_locator(MultipleLocator(step))

    # axs.grid(axis="both", which='major', linestyle='-')
    # axs.grid(axis="both", which='minor', linestyle='-')

    # axs.set_ylim(-step, 1)
    axs.set_yticks([0, 0.5, 1])
    axs.set_xlabel('score')
    axs.set_ylabel('metric value')
    
#     ymin = int(step * np.floor(min(prec.min(), rec.min())/step))
#     ymax = int(step * np.ceil(max(prec.max(), rec.max())/step))
    
#     if __r < rows - 1:
#         axs.set_xticklabels([])
#     else:
#         axs.set_xlabel('Threshold')
#     if __c != 0:
#         axs.set_yticklabels([])
#     else:
#         axs.set_ylabel('Value')
#    axs.grid()
#    axs.legend()

    return axs


def f1scores(pre, rec):
    '''`f1scores(pre, rec)` returns a numpy array of the f1scores calculated
    element-wise for each precision and recall'''
    f1_ = np.zeros(len(pre), dtype=float)
    isfinite = (pre != 0) & (rec != 0)
    f1_[isfinite] = hmean(np.array([pre[isfinite], rec[isfinite]]), axis=0)
    return f1_


def grid4plot(numaxes, ncols=None):
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


if __name__ == "__main__":
    dfr = pd.DataFrame({'mag': [1, 1, 3, 5, 10], 'dist': [0, 1, 0, 1, 0]})
    ret = heatmap_df(dfr, 'mag', 'dist', np.arange(5), np.arange(5))
    pd.set_option('max_rows', 200)
    print(ret.to_string())
# printdoc()

# EVALPATH HIDDEN
# REMOVE show from plots
