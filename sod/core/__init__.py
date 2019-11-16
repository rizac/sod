import sys
import pandas as pd

if sys.version_info[0] < 3 or sys.version_info[1] < 7:
    from collections import OrderedDict as odict
else:
    odict = dict  # pylint: disable=invalid-name


def pdconcat(dataframes, **kwargs):
    '''forwards to pandas concat with standard arguments'''
    return pd.concat(dataframes, sort=False, axis=0, copy=True, **kwargs)
