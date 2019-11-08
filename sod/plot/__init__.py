import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sod.evaluation import is_outlier, CLASSES, pdconcat
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
from itertools import product, repeat, cycle
# %matplotlib inline

PLOT_RATIO = 0.2  # there is too much data, show only this ratio


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
    # divide the dataframe in bins. Take PLOT_RATIO randomly points for each bin
    for name, fff in CLASSES.items():
        class_df = df[fff(df)]
        if class_df.empty:
            continue
        qcuts = [pd.qcut(class_df[k], 100, duplicates='drop') for k in cols]
        # make a series with bins (one for each column) -> num of pts per bins:
        series_df_bins = class_df.groupby(qcuts).size()
        # convert to dataframe with columns:
        class_df_bins = series_df_bins.reset_index()
        # column 0 is the columns with the counts (how many pts per interval)
        # log-normalize it:
        class_df_bins[0] = np.log10(1 + 9*class_df_bins[0]/np.nanmax(class_df_bins[0]))
        # drop zeros:
        class_df_bins = class_df_bins[class_df_bins[0] > 0]
        # convert bins to midpoints:
        for col in cols:
            class_df_bins[col] = class_df_bins[col].apply(lambda val: (val.left+val.right)/2.0)
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
        if col_z is None:
            ax.scatter(df[col_x], df[col_y], color=colors, **kwargs)
        else:
            ax.scatter(df[col_x], df[col_y], df[col_z], color=colors, **kwargs)
        return ax

    colors = [
        [0, 0.1, 0.75],
        [0.75, 0.1, 0],
        [0, 0.75, 0.1]
    ]
    for _ in range(len(CLASSES)-len(colors)):
        colors.append([0.75, 0.5, 0])

    for i, ((name, _df_), color) in enumerate(zip(dfs.items(), colors)):
        ax_ = scatter(newaxes(i+1), _df_, color)
        ax_.set_title('%s: %d segments' % (name, len(_df_)))

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

    wspace = .4 if col_z is not None else .25
    hspace = wspace
    if clfs is not None and col_z is None:
        hspace *= len(clfs)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=wspace, hspace=hspace)
    return fig


# def plot_decision_func_2d(axs, clf):
#     x0, x1 = axs.get_xlim()
#     y0, y1 = axs.get_ylim()
#     print([x0,x1, y0, y1])
#     xx, yy = np.meshgrid(np.linspace(x0, y1, 101, endpoint=True),
#                          np.linspace(y0, y1, 101, endpoint=True))
#     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     print(Z)
# 
#     # print contours shaded (ignore it is just for showing off):
#     # plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
# 
#     # print contour:
#     axs.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
# 
# def _make_contour(clf, xxx, yyy):
#     zzz = clf.decision_function(np.c_[xxx.ravel(), yyy.ravel()])
#     return zzz.reshape(xxx.shape)
    