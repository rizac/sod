import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sod.evaluation import is_outlier, CLASSES, pdconcat
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
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
        _dfs = []
        for kol in cols:
            bins = pd.cut(class_df[kol], 10)
            for _, df_ in class_df.groupby(bins):
                if df_.empty:
                    continue
                num = max(1, int(len(df_) * PLOT_RATIO))
                _dfs.append(df_.sample(num)[cols])

        dfs[name] = pdconcat(_dfs)

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
        kwargs = {'edgecolors': 'none', 's': 81}
        if col_z is None:
            ax.scatter(df[col_x], df[col_y], color=color, **kwargs)
        else:
            ax.scatter(df[col_x], df[col_y], df[col_z], color=color, **kwargs)
        return ax

    alpha = 0.01
    colors = [
        [0, 0.1, 0.75, alpha],
        [0.75, 0.1, 0, alpha],
        [0, 0.75, 0.1, alpha]
    ]
    for _ in range(len(CLASSES)-len(colors)):
        colors.append([0.75, 0.5, 0, alpha])

    for i, ((name, _df_), color) in enumerate(zip(dfs.items(), colors)):
        ax_ = scatter(newaxes(i+1), _df_, color)
        ax_.set_title('%s: %d segments' % (name, len(_df_)))

    if clfs is not None and col_z is None:
        xxx, yyy = np.meshgrid(np.linspace(minx, maxx, 100, endpoint=True),
                               np.linspace(miny, maxy, 100, endpoint=True))
        for name, clf in clfs.items():
            zzz = clf.decision_function(np.c_[xxx.ravel(), yyy.ravel()])
            zzz = zzz.reshape(xxx.shape)
            for axs in fig.axes:
                axs.contour(xxx, yyy, zzz, levels=[0], linewidths=2, colors='darkred')

        plt.legend()
    space = .4 if col_z is not None else .25
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=space, hspace=space)
    return fig


def plot_decision_func_2d(axs, clf):
    x0, x1 = axs.get_xlim()
    y0, y1 = axs.get_ylim()
    print([x0,x1, y0, y1])
    xx, yy = np.meshgrid(np.linspace(x0, y1, 101, endpoint=True),
                         np.linspace(y0, y1, 101, endpoint=True))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    print(Z)

    # print contours shaded (ignore it is just for showing off):
    # plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)

    # print contour:
    axs.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')

def _make_contour(clf, xxx, yyy):
    zzz = clf.decision_function(np.c_[xxx.ravel(), yyy.ravel()])
    return zzz.reshape(xxx.shape)
    