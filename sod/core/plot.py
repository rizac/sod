'''plot utilities'''

import pandas as pd
import numpy as np
from itertools import product, cycle
import warnings
from math import sqrt
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from sklearn.calibration import calibration_curve  # , CalibratedClassifierCV

from sod.core.dataset import dataset_info, floatingcols, OUTLIER_COL, is_outlier
# from sod.core.evaluation import is_outlier, pdconcat

# %matplotlib inline


def plotdist(df, columns=None, bins=20, axis_lim=None, class_indices=None):
    '''
    Plots the distributions of all `columns` of df
    '''
    if columns is None:
        columns = list(floatingcols(df))
    dinfo = _dataset_info(df)

    classnames = dinfo.classnames
    if class_indices is not None:
        classnames = [_ for i, _ in enumerate(classnames)
                      if i in class_indices]

    rows, cols = len(columns), len(classnames)
    fig = plt.figure(figsize=(15, 15))

    index = 1
    # divide the dataframe in bins using pandas qcut, which creates bins
    # with roughly the same size of samples per bin
    for col in columns:
        # pandas min and max skipna by default:
        if axis_lim is None:
            min_, max_ = (df[col]).min(), (df[col]).max()
        else:
            min_, max_ = df[col].quantile([1-axis_lim, axis_lim])
        bins_ = np.linspace(min_, max_, bins, endpoint=True)
        # plot one row of subplots:
        for cls_index, cname in enumerate(classnames):
            ax = fig.add_subplot(rows, cols, index)
            index += 1
            class_df = df[dinfo.class_selector[cname]][col]
#             qcut = pd.cut(class_df, bins, include_lowest=True,
#                           duplicates='drop')
#             # make a series with bins (one for each column) -> num of pts per bins:
#             series_df_bins = class_df.groupby(qcut).size()
            # plot histogram
            ax.hist(class_df, bins_)
            ax.set_xlabel(cname)
            if cls_index == 0:
                ax.set_ylabel(str(col))
            ax.grid(True)
            # ax.set_yscale('log')

    wspace = .5  # if col_z is not None else .25
    hspace = wspace
#     if clfs is not None and col_z is None:
#         hspace *= len(clfs)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=wspace, hspace=hspace)
    return fig


def _get_grid(num):
    '''returns the tuple (row, col) of integers denoting the best
    layout for the given `num` (denoting the number of plots to display)
    '''
    if num < 4:
        return 3, 1
    rows = int(sqrt(num))
    cols = rows
    if rows * cols < num:
        rows += 1
    if rows * cols < num:
        cols += 1
    return rows, cols


def _dataset_info(df):
    '''Returns the dataset info of the given dataframe `df` or a simple
    newly created dataset info splitting according to the column
    `OUTLIER_COL` ('outlier'). If such a column does not exist, this function
    raises Exception
    '''
    try:
        return dataset_info(df)
    except Exception:
        msg = "Dataframe is not bound to any implemented dataset"
        if OUTLIER_COL not in df.columns:
            raise Exception(msg + ", nor it has a column named \"%s\": "
                            "can not plot" % OUTLIER_COL)
        warnings.warn(msg + ", splitting classes based on the"
                      "\"%s\" column" % OUTLIER_COL)

        class dinfo:

            classnames = ['ok', OUTLIER_COL]

            class_selector = {
                'ok': lambda d: ~is_outlier(d),
                OUTLIER_COL: is_outlier
            }

        return dinfo


def plot(df, col_x, col_y, col_z=None, axis_lim=None, clfs=None,
         class_indices=None):
    '''Plots the given dataframe per classes. The dataframe must have been
    implemented as Dataset (see module `dataset`) OR have at least the
    boolean column 'outlier', denoting whether the rows are outliers or
    inliers.

    :param axis_lim: float in [0, 1] or None, is the quantile of data to be
    shown on the axis: 0.95 will display the axis min and max at 0.05
    quantile of the data distribution and 0.95 quantile, respectuvely

    :param class_indices: the indices of the classes to display, wrt the
        passed dataframe's dataset. None (the default) will show all classes
    '''

    cols = [col_x, col_y] + ([] if col_z is None else [col_z])
    df = df.dropna(subset=cols)

    if axis_lim is None:
        # pandas min and max skipna by default:
        minx, maxx = (df[col_x]).min(), (df[col_x]).max()
        miny, maxy = (df[col_y]).min(), (df[col_y]).max()
        minz, maxz = (None, None)
        if col_z is not None:
            miny, maxy = (df[col_z]).min(), (df[col_z]).max()
    else:
        minx, maxx = df[col_x].quantile([1-axis_lim, axis_lim])
        miny, maxy = df[col_y].quantile([1-axis_lim, axis_lim])
        minz, maxz = (None, None)
        if col_z is not None:
            minz, maxz = df[col_z].quantile([1-axis_lim, axis_lim])

    dfs = {}
    numsegments = {}
    dinfo = _dataset_info(df)
    classnames = dinfo.classnames
    if class_indices is not None:
        classnames = [_ for i, _ in enumerate(classnames)
                      if i in class_indices]

    # colors for the points. Do not specify a fourth element
    # (alpha=transparency)
    # as it will be added according to the points density
    OUTLIER_COLOR = [0.75, 0.5, 0]
    INLIER_COLOR = [0, 0.1, 0.75]
    # other colors:
    # red: [0.75, 0.1, 0]
    # green [0, 0.75, 0.1]
    # yellow/brown: [0.75, 0.5, 0]

    classcolors = {}
    # divide the dataframe in bins using pandas qcut, which creates bins
    # with roughly the same size of samples per bin
    for name in classnames:

        fff = dinfo.class_selector[name]

        class_df = df[fff(df)]

        numsegments[name] = len(class_df)
        if class_df.empty:
            continue

        classcolors[name] = OUTLIER_COLOR if class_df[OUTLIER_COL].values[0] \
            else INLIER_COLOR

        if col_z is None and axis_lim is not None:
            # remove out of bounds so that the granularity of bins is
            # zoom  dependent (otherwise too big bins <-> rectangles for
            # dataframes with far far away outliers):
            class_df = class_df[class_df[col_x].between(minx, maxx)]
            class_df = class_df[class_df[col_y].between(miny, maxy)]

        qcuts = [pd.cut(class_df[k], 100, duplicates='drop') for k in cols]
        # make a series with bins (one for each column) -> num of pts per bins:
        series_df_bins = class_df.groupby(qcuts).size()
        # convert to dataframe with columns:
        class_df_bins = series_df_bins.reset_index()
        # column 0 is the columns with the counts (how many pts per interval)
        # log-normalize it:
        class_df_bins[0] = \
            np.log10(1.5 + 8.5*class_df_bins[0]/np.nanmax(class_df_bins[0]))
        # drop zeros:
        class_df_bins = class_df_bins[class_df_bins[0] > 0]
        # convert bins to midpoints:
        # but only in case of Axes3D. In 2d mode, we will draw rectangles
        if col_z is not None:
            for col in cols:
                class_df_bins[col] = \
                    class_df_bins[col].apply(lambda val: val.mid)
        dfs[name] = class_df_bins

    fig = plt.figure()  # figsize=(15, 15))
    rows, cols = _get_grid(len(classnames))

    def newaxes(index):
        if col_z is not None:
            ax = fig.add_subplot(rows, cols, index, projection='3d')
        else:
            ax = fig.add_subplot(rows, cols, index)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        if col_z is not None:
            ax.set_zlabel(col_z)
            ax.set_zlim(minz, maxz)
        ax.grid(True)
        return ax

    def scatter(ax, df, basecolor):
        kwargs = {'edgecolors': 'none', 's': 64}
        colors = np.zeros(shape=(len(df), 4))
        colors[:, 0] = basecolor[0]
        colors[:, 1] = basecolor[1]
        colors[:, 2] = basecolor[2]
        colors[:, 3] = df[0].values
        if col_z is not None:  # 3D
            ax.scatter(df[col_x], df[col_y], color=colors, **kwargs)
        else:
            rects = [
                Rectangle((c_x.left, c_y.left), c_x.length, c_y.length)
                for c_x, c_y in zip(df[col_x], df[col_y])
            ]
            ax.add_collection(
                PatchCollection(rects, facecolors=colors, edgecolors='None',
                                linewidth=0)
            )
        return ax

    for i, (name, _df_) in enumerate(dfs.items()):
        ax_ = scatter(newaxes(i+1), _df_, classcolors[name])
        ax_.set_title('%s: %d segments' % (name, numsegments[name]))

    # draw each classifier decision function's contour:
    if clfs is not None and col_z is None:
        clst = product(['r', 'b', 'g', 'y', 'c', 'm'],
                       ['solid', 'dashed', 'dashdot', 'dotted'])
        xxx, yyy = np.meshgrid(np.linspace(minx, maxx, 100, endpoint=True),
                               np.linspace(miny, maxy, 100, endpoint=True))
        for (name, clf), (color, linestyle) in zip(clfs.items(), cycle(clst)):
            zzz = clf.decision_function(np.c_[xxx.ravel(), yyy.ravel()])
            zzz = zzz.reshape(xxx.shape)
            for axs in fig.axes:
                axs.contour(xxx, yyy, zzz, levels=[0], linewidths=2,
                            colors=color, linestyles=linestyle)
                ttt = axs.get_title()
                ttt += "\nclassifier '%s': color '%s', %s" % (name, color, linestyle)
                axs.set_title(ttt)

    # set subplots spaces:
    wspace = .2  # if col_z is not None else .25
    hspace = wspace
    if clfs is not None and col_z is None:
        hspace *= (1 + len(clfs))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=wspace, hspace=hspace)
    return fig


def plot_calibration_curve(estimators, test_df, columns):
    """Plot calibration curve for est w/o and with calibration.

    :param estimators: dict of strings names mapped to a fitted classifier
    """
    X_test = test_df[columns].values

    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for name, clf in estimators.items():
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            min_, max_ = np.nanmin(prob_pos), np.nanmax(prob_pos)
            prob_pos = (prob_pos - min_) / (max_ - min_)

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(test_df['outlier'].astype(int),
                              prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label=name)

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()