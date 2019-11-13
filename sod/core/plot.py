import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
from itertools import product, repeat, cycle
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from sod.core.evaluation import is_outlier, CLASSES, pdconcat
# %matplotlib inline


def plot(df, col_x, col_y, col_z=None, axis_lim=None, clfs=None):
    '''axis_lim is the quantile of data to be shown on the axis: 0.95 will
    display the axis min and max at 0.05 quantile of the data distribution
    and 0.95 quantile, respectuvely'''

    cols = [col_x, col_y] + ([] if col_z is None else [col_z])
    df = df.dropna(subset=cols)

    if axis_lim is None:
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
    # divide the dataframe in bins using pandas qcut, which creates bins
    # with roughly the same size of samples per bin
    for name, fff in CLASSES.items():
        class_df = df[fff(df)]

        numsegments[name] = len(class_df)
        if class_df.empty:
            continue

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

    fig = plt.figure(figsize=(15, 15))
    rows = int(sqrt(len(CLASSES)))
    cols = rows
    if rows * cols < len(CLASSES):
        rows += 1
    if rows * cols < len(CLASSES):
        cols += 1

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

    def scatter(ax, df, color):
        kwargs = {'edgecolors': 'none', 's': 64}
        colors = np.zeros(shape=(len(df), 4))
        colors[:, 0] = color[0]
        colors[:, 1] = color[1]
        colors[:, 2] = color[2]
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

    # create colors:
    colors = [
        [0, 0.1, 0.75],  # segments ok
        [0.75, 0.1, 0],  # wrong inv
        [0, 0.75, 0.1]   # swap acc <-> vel
    ]
    # now for all other classes set the same color:
    for _ in range(len(CLASSES)-len(colors)):
        colors.append([0.75, 0.5, 0])

    for i, ((name, _df_), color) in enumerate(zip(dfs.items(), colors)):
        ax_ = scatter(newaxes(i+1), _df_, color)
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
    wspace = .4 if col_z is not None else .25
    hspace = wspace
    if clfs is not None and col_z is None:
        hspace *= len(clfs)
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