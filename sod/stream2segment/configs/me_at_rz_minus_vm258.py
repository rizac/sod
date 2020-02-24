'''
Stream2segment module for processing the data and create our dataset whreby to train
and test our classifier
'''
from __future__ import division

# # make the following(s) behave like python3 counterparts if running from python2.7.x
# # (http://python-future.org/imports.html#explicit-imports):
# from builtins import (ascii, bytes, chr, dict, filter, hex, input,
#                       int, map, next, oct, open, pow, range, round,
#                       str, super, zip)
import os
import sys
# import math

# OrderedDict is a python dict that returns its keys in the order they are inserted
# (a normal python dict returns its keys in arbitrary order)
# Useful e.g. in  "main" if we want to control the *order* of the columns in the output csv
from collections import OrderedDict
# from datetime import datetime, timedelta  # always useful
# from math import factorial  # for savitzky_golay function

# import numpy for efficient computation:
import numpy as np
import pandas as pd
# import obspy core classes (when working with times, use obspy UTCDateTime when possible):
# from obspy import Trace, Stream, UTCDateTime
from obspy.signal.spectral_estimation import PPSD
# from obspy.core.inventory.inventory import read_inventory
# from obspy.geodetics.base import degrees2kilometers
# from obspy.geodetics import degrees2kilometers as d2km
# decorators needed to setup this module @gui.sideplot, @gui.preprocess @gui.customplot:
# from stream2segment.process import gui
# strem2segment functions for processing obspy Traces. This is just a list of possible functions
# to show how to import them:
# from stream2segment.process.math.traces import ampratio, bandpass, cumsumsq,\
#     timeswhere, fft, maxabs, utcdatetime, ampspec, powspec, timeof, sn_split
# stream2segment function for processing numpy arrays:
# from stream2segment.process.math.ndarrays import triangsmooth, snr
# from stream2segment.process.db import get_inventory
from stream2segment.process.math.traces import ampratio, utcdatetime

HANDLABELLED_COL = 'hand_labelled'
OUTLIER_COL = 'outlier'
WINDOWTYPE_COL = 'window_type'


# IN PRINCIPLE, FOR EACH DATABASE YOU SHOULD ONLY OVERRIDE THIS
# AND CHANGE THE CONFIG ACCORDING TO YOUR NEEDS:
def get_traces_from_segment(segment, config):
    '''
    THIS METHOD IS THE FIRST AS IT CAN/SHOULD BE OVERRIDDEN AND EVERYTHING
    ELSE LEFT AS IT IS.

    IN THIS METHOD THE USER SHOULD RETURN THE TRACES TO BE PROCESSED FROM
    THE GIVEN SEGMENT AND THE CONFIG PARAMETERS (except `sn_windows` which is
    handled automatically and should **NOT** be used here).

    You should return an array (list, tuple) of elements:
    [
        (trace, inventory, param_dicts)
        ...
        (trace, inventory, param_dicts)
    ]
    from the given segment. By default, it returns a signle trace from the
    given segment:
    [
        (trace, inv, {})
    ]

    NOTES: In params the user should put every parameter which should be
    overwritten. There are in particular 2 parameters which might be dependent
    on what is implemented here and are:
    {
        HANDLABELLED_COL: False,
        OUTLIER_COL: False
    }
    Thus, you can overwrite the parameter like this:
    [
        (trace1, inv, {HANDLABELLED_COL: True}),
        ...
    ]
    '''
    tra, inv = segment.stream()[0], segment.inventory()
    params = {  # NOTE when OVERWRITING: THESE keys MUST BE ALWAYS PRESENT:
        HANDLABELLED_COL: False,
        OUTLIER_COL: False
    }
    # overwritten stuff:
    if segment.station.id in config['bad']:
        params[OUTLIER_COL] = True
        params[HANDLABELLED_COL] = True
    elif segment.station.id in config['good']:
        params[OUTLIER_COL] = False
        params[HANDLABELLED_COL] = True

    return [(tra, inv, params)]


def assert1trace(stream):
    '''asserts the stream has only one trace, raising an Exception if it's not the case,
    as this is the pre-condition for all processing functions implemented here.
    Note that, due to the way we download data, a stream with more than one trace his
    most likely due to gaps / overlaps'''
    # stream.get_gaps() is slower as it does more than checking the stream length
    if len(stream) != 1:
        raise ValueError("%d traces (probably gaps/overlaps)" % len(stream))


def main(segment, config):
    '''main function'''
    # annoying print statementin obspy 1.1.1  when calling the function 'psd'
    # (see below) and when adding a trace shorter than
    # the ppsd_length: workaround? redirect to stderr (which is captured by the caller):
    temp = sys.stdout
    try:
        sys.stdout = sys.stderr

        stream = segment.stream(True)
        assert1trace(stream)  # raise and return if stream has more than one trace
        # raw_trace = stream[0]

        ret = get_psd_values_df(segment, config)
        # pd.concat(data, sort=False, ignore_index=True, copy=True, axis=0)

        ret['amplitude_ratio'] = ampratio(segment.stream()[0])
        ret['event_id'] = segment.event_id
        ret['station_id'] = segment.station.id
        ret['event_time'] = segment.event.time
        # store channel's channel via data_seed_id (might be faster):
        # calculate it here so in case of error we avoid unnecessary calculations
        net_sta_loc_cha = segment.data_seed_id.split('.')
        ret['location_code'] = net_sta_loc_cha[2]
        ret['channel_code'] = net_sta_loc_cha[3]
        ret['magnitude'] = segment.event.magnitude
        ret['distance_km'] = segment.event_distance_km
        ret['dataset_id'] = config['dataset_id']
        return ret
    finally:
        sys.stdout = temp


def get_psd_values_df(segment, config):
    """
    Gets the PSD values in form of DataFrame. Does not check if
    the stream has gaps oiverlaps (assumes it has not).
    Checks in the config if the psd values should be calculated on sub windows
    of segment.stream() or not (parameter 'sn_windows')

    :param inventory: if None, uses the segment inventory.
    """
    traces_invs_params = get_traces_from_segment(segment, config)

    sn_wdw = config['sn_windows']
    wlen_sec = sn_wdw['signal_window']
    if wlen_sec:
        atime = utcdatetime(segment.arrival_time) + sn_wdw['arrival_time_shift']
        new_traces_invs_params = []
        for (tra, inv, params) in traces_invs_params:
            noi_wdw = [tra.slice(None, atime), inv,
                       {**params, WINDOWTYPE_COL: False}]
            # window_type True: has signal, False: is noise
            sig_wdw = [tra.slice(atime, None), inv,
                       {**params, WINDOWTYPE_COL: True}]
            # window_type True: has signal, False: is noise
            new_traces_invs_params.extend([noi_wdw, sig_wdw])
        traces_invs_params = new_traces_invs_params
    else:
        # simply set window_type param, overwriting any setting is present:
        for (tra, inv, params) in traces_invs_params:
            # window_type True: has signal, False: is noise
            params[WINDOWTYPE_COL] = True

    ret_dfs = []
    required_psd_periods = config['psd_periods']
    for tra, inv, params in traces_invs_params:

        # PSD NOISE VALUES:
        required_psd_values = psd_values(required_psd_periods, tra,
                                         segment.inventory()
                                         if inv is None else inv)

        # calculates amplitudes at the frequency bins given in the config file:

        # write stuff to csv:
        ret = OrderedDict()

        for period, psdval in zip(required_psd_periods, required_psd_values):
            ret['psd@%ssec' % str(period)] = float(psdval)

        ret = {
            **params,
            'length_sec': tra.stats.endtime - tra.stats.starttime,  # <- float
            **ret
        }

        # # Here we tried to programmatically label artifacts as outliers
        # # But we realised later that the lines below have no effect as
        # # they should be executed BEFORE the creation of `ret` above.
        # # We also realised that it is better to handle these artifacts later
        # # in a Jupyter notebook and put them in a specified data frame.
        # # So, all in all, let's comment them out:
        # if (required_psd_values[~np.isnan(required_psd_values)] <= -1000).all():
        #    params[HANDLABELLED_COL] = True
        #    params[OUTLIER_COL] = True

        ret_dfs.append(ret)

    return pd.DataFrame(ret_dfs)


def psd_values(periods, raw_trace, inventory):
    periods = np.asarray(periods)
    try:
        ppsd_ = psd(raw_trace, inventory)
    except Exception as esc:
        raise ValueError('%s error when computing PSD: %s' %
                         (esc.__class__.__name__, str(esc)))
    # check first if we can interpolate ESPECIALLY TO SUPPRESS A WEIRD
    # PRINTOUT (numpy?): something like '5064 5062' which happens
    # on IndexError (len(ppsd_.psd_values)=0)
    if not len(ppsd_.psd_values):
        raise ValueError('Expected 1 psd array, no psd computed')
    val = np.interp(
        np.log10(periods),
        np.log10(ppsd_.period_bin_centers),
        ppsd_.psd_values[0]
    )
    val[periods < ppsd_.period_bin_centers[0]] = np.nan
    val[periods > ppsd_.period_bin_centers[-1]] = np.nan
    return val


def psd(raw_trace, inventory):
    # tr = segment.stream(True)[0]
    dt = raw_trace.stats.endtime - raw_trace.stats.starttime  # total_seconds
    ppsd = PPSD(raw_trace.stats, metadata=inventory, ppsd_length=int(dt))
    ppsd.add(raw_trace)
    return ppsd
