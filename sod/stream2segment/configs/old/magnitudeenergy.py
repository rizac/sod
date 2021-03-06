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
import math

import pandas as pd
# OrderedDict is a python dict that returns its keys in the order they are inserted
# (a normal python dict returns its keys in arbitrary order)
# Useful e.g. in  "main" if we want to control the *order* of the columns in the output csv
from collections import OrderedDict
from datetime import datetime, timedelta  # always useful
from math import factorial  # for savitzky_golay function

# import numpy for efficient computation:
import numpy as np
# import obspy core classes (when working with times, use obspy UTCDateTime when possible):
from obspy import Trace, Stream, UTCDateTime
from obspy.signal.spectral_estimation import PPSD
from obspy.core.inventory.inventory import read_inventory
from obspy.geodetics.base import degrees2kilometers
# from obspy.geodetics import degrees2kilometers as d2km
# decorators needed to setup this module @gui.sideplot, @gui.preprocess @gui.customplot:
from stream2segment.process import gui
# strem2segment functions for processing obspy Traces. This is just a list of possible functions
# to show how to import them:
from stream2segment.process.math.traces import ampratio, bandpass, cumsumsq,\
    timeswhere, fft, maxabs, utcdatetime, ampspec, powspec, timeof, sn_split
# stream2segment function for processing numpy arrays:
from stream2segment.process.math.ndarrays import triangsmooth, snr
from stream2segment.process.db import get_inventory


def assert1trace(stream):
    '''asserts the stream has only one trace, raising an Exception if it's not the case,
    as this is the pre-condition for all processing functions implemented here.
    Note that, due to the way we download data, a stream with more than one trace his
    most likely due to gaps / overlaps'''
    # stream.get_gaps() is slower as it does more than checking the stream length
    if len(stream) != 1:
        raise ValueError("%d traces (probably gaps/overlaps)" % len(stream))


def main(segment, config):
    '''main function: wraps main2 and raises ValueError(s)
    in order NOT to stop the whole processing but the currently processed segment
    only'''
    try:
        return main2(segment, config)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(str(exc))


def main2(segment, config):
    '''calls _main with the normal inventory and all possible
    wrong inventories'''
    
    # annoying print statementin obspy 1.1.1  when calling the function 'psd'
    # (see below) and when adding a trace shorter than
    # the ppsd_length: workaround? redirect to stderr (which is captured by the caller):
    temp = sys.stdout
    try:
        sys.stdout = sys.stderr

        stream = segment.stream(True)
        assert1trace(stream)  # raise and return if stream has more than one trace
        raw_trace = stream[0].copy()

        # store channel's channel via data_seed_id (might be faster):
        # calculate it here so in case of error we avoid unnecessary calculations
        channel_code = segment.data_seed_id.split('.')[3]
    
        # compute amplitude ratio only once on the raw trace:
        amp_ratio = ampratio(raw_trace)
#         if amp_ratio >= config['amp_ratio_threshold']:
#             saturated = True  # @UnusedVariable

        # data = []
        # bandpass the trace, according to the event magnitude.
        # This modifies the segment.stream() permanently:
        # data.append(_main(segment, config, raw_trace, segment.inventory()))

        # add gain (1). Decide whether to compute gain x2,10,100 or x1/2.1/10,1/100
        # (not both, try to speed up a bit the computations)
        # "randomly" according to the segment id (even or odd)
#         gain_factors = config['stage_gain_factors']
#         if segment.id % 2 == 0:
#             gain_factors = gain_factors[:3]
#         else:
#             gain_factors = gain_factors[3:]
#         for gain_factor in gain_factors:
#             # need to change alkso the raw trace for the noisepsd, otherwise
#             # we calculate the same psd as if we did not change the gain:
#             raw_trace_ = raw_trace.copy()
#             raw_trace_dtype_ = raw_trace_.data.dtype
#             raw_trace_.data = raw_trace_.data * float(gain_factor)
#             if np.issubdtype(raw_trace_dtype_, np.integer):
#                 raw_trace_.data = (0.5 + raw_trace_.data).astype(raw_trace_dtype_)
#             data.append(_main(segment, config, raw_trace_, segment.inventory()))
#             data[-1]['outlier'] = True
#             data[-1]['modified'] = "STAGEGAIN:X%s" % str(gain_factor)
# 
#         # acceleromters/velocimeters:
#         if segment.station.id in config['station_ids_both_accel_veloc']:
#             # reload inventory (takes more time, but we won't modify cached version):
#             inventory = get_inventory(segment.station)
#             cha_obj = get_cha_obj(segment, inventory)
#             resp_tmp = cha_obj.response
#             for other_cha in get_other_chan_objs(segment, inventory):
#                 cha_obj.response = other_cha.response
#                 data.append(_main(segment, config, raw_trace, inventory))
#                 data[-1]['outlier'] = True
#                 data[-1]['modified'] = "CHARESP:%s" % other_cha.code
#             cha_obj.response = resp_tmp
# 
#         if segment.station.id in config['station_ids_with_wrong_local_inventory']:
#             channels_ = config['station_ids_with_wrong_local_inventory'][segment.station.id]
#             filename = channels_.get(segment.data_seed_id, None)
#             if filename is None:
#                 raise ValueError('%s not found in wrong inventories dict' % segment.data_seed_id)
#             if filename is not None:
#                 inventories_dir = config['inventories_dir']
#                 wrong_inventory = read_inventory(os.path.join(os.getcwd(), inventories_dir,
#                                                               filename))
#                 data.append(_main(segment, config, raw_trace, wrong_inventory))
#                 data[-1]['outlier'] = True
#                 data[-1]['modified'] = "INVFILE:%s" % filename

        ret = _main(segment, config, raw_trace, segment.inventory())
        # data[-1]['outlier'] = True

        # ret = pd.concat(data, sort=False, ignore_index=True, copy=True, axis=0)
        ret['amplitude_ratio'] = amp_ratio
        ret['event_id'] = segment.event_id
        ret['station_id'] = segment.station.id
        ret['event_time'] = segment.event.time
        ret['channel_code'] = channel_code
        ret['magnitude'] = segment.event.magnitude
        ret['distance_km'] = segment.event_distance_km
        return ret

    finally:
        sys.stdout = temp


def get_cha_obj(segment, inventory=None):
    '''Returns the obspy channel object of the given segment'''
    if inventory is None:
        inventory = segment.inventory()
    for n, net in enumerate(inventory):
        if net.code == segment.station.network:
            for s, sta in enumerate(net):
                if sta.code == segment.station.station:
                    for c, cha in enumerate(sta):
                        if cha.code == segment.channel.channel:
                            return cha
    return None


def is_accelerometer(segment):
    # we might use channel code but we want to call _is_accelerometer:
    return _is_accelerometer(segment.channel.channel)


def _is_accelerometer(channel_code):
    return channel_code[1:2].lower() in ('n', 'l', 'g')


def get_other_chan_objs(segment, inventory=None):
    '''Yields channels objects within the given inventory'''
    if inventory is None:
        inventory = segment.inventory()
    is_accel = is_accelerometer(segment)
    for n, net in enumerate(inventory):
        for s, sta in enumerate(net):
            for c, cha in enumerate(sta):
                if _is_accelerometer(cha.code) != is_accel:
                    # print('channel code: %s, other channel code: %s' % (segment.channel.channel, cha.code))
                    yield cha


def _main(segment, config, raw_trace, inventory_used):
    """
    called by main with supplied inventory_used, which MUST be the inventory used
    on the raw trace to obtain `segment.stream()[0]`
    """
    required_psd_periods = config['psd_periods']

#     atime = utcdatetime(segment.arrival_time)

    # trace.slice calls trace.copy().trim:
#     traces = [
#         (raw_trace.slice(atime-60, atime), 'n'),
#         (raw_trace.slice(atime-30, atime+30), 'ns'),
#         (raw_trace.slice(atime, atime+60), 's')
#     ]
    

#     ret_df = []

#    for raw_trace, window_type in traces:
#         # cumulative of squares:
#         cum_labels = [0.05, 0.95]
#         cum_trace = cumsumsq(trace, normalize=True, copy=True)
#         cum_times = timeswhere(cum_trace, *cum_labels)
#     
#         # Caluclate PGA and PGV
#         # FIXME! THERE IS AN ERROR HERE WE SHOULD ITNEGRATE ONLY IF WE HAVE AN
#         # ACCELEROMETER! ISN't IT?
#         t_PGA, PGA = maxabs(trace, cum_times[0], cum_times[-1])
#         trace_int = trace.copy().integrate()
#         t_PGV, PGV = maxabs(trace_int, cum_times[0], cum_times[-1])
#     
#         # CALCULATE SPECTRA (SIGNAL and NOISE)
#         spectra = _sn_spectra(segment, config)
#         normal_f0, normal_df, normal_spe = spectra['Signal']
#         noise_f0, noise_df, noise_spe = spectra['Noise']  # @UnusedVariable
#     
#         # AMPLITUDE (or POWER) SPECTRA VALUES and FREQUENCIES:
#         required_freqs = config['freqs_interp']
#         ampspec_freqs = normal_f0 + normal_df * np.arange(len(normal_spe))
#         required_amplitudes = np.interp(np.log10(required_freqs),
#                                         np.log10(ampspec_freqs),
#                                         normal_spe) / segment.sample_rate
#     
#         # SNR:
#         
#         fcmin = mag2freq(magnitude)
#         fcmax = config['preprocess']['bandpass_freq_max']  # used in bandpass_remresp
#         spectrum_type = config['sn_spectra']['type']
#         snr_ = snr(normal_spe, noise_spe, signals_form=spectrum_type,
#                    fmin=fcmin, fmax=fcmax, delta_signal=normal_df, delta_noise=noise_df)
    
        # PSD NOISE VALUES:
        # FIXME! DO I HAVE TO PASS THE PROCESSED TRACE (AS IT IS) or THE RAW ONE
        # (segment.stream(True)[0])?
        
    required_psd_values = psd_values(required_psd_periods,
                                     raw_trace,
                                     inventory_used)

    # calculates amplitudes at the frequency bins given in the config file:

    # write stuff to csv:
    ret = OrderedDict()

    for period, psdval in zip(required_psd_periods, required_psd_values):
        ret['psd@%ssec' % str(period)] = float(psdval)

    if segment.station.id in config['bad']:
        ret['outlier'] = True
        ret['subclass'] = ''
    elif segment.station.id in config['good']:
        ret['outlier'] = False
        ret['subclass'] = ''
    elif segment.station.id in config['suspect']:
        ret['outlier'] = False
        ret['subclass'] = 'unlabeled.maybe.outlier'
    else:
        ret['outlier'] = False
        ret['subclass'] = 'unlabeled.unknown'

        # ret['modified'] = ''
        # ret['window_type'] = window_type
        # ret['start_time'] = raw_trace.stats.starttime.datetime
        # ret['length_sec'] = raw_trace.stats.endtime - raw_trace.stats.starttime

#         ret_df.append(ret)
# 
#     return pd.DataFrame(ret)
    return ret


def __bandpass_remresp(segment, config, trace, inventory):
    """Applies a pre-process on the given segment waveform by
    filtering the signal and removing the instrumental response.
    DOES modify the segment stream in-place (see below)

    The filter algorithm has the following steps:
    1. Sets the max frequency to 0.9 of the Nyquist frequency (sampling rate /2)
    (slightly less than Nyquist seems to avoid artifacts)
    2. Offset removal (subtract the mean from the signal)
    3. Tapering
    . Pad data with zeros at the END in order to accommodate the filter transient
nf['bandpass_freq_max']
    5. Apply bandpass filter, where the lower frequency is set according to the magnitude
    6. Remove padded elements
    7. Remove the instrumental response

    IMPORTANT NOTES:
    - Being decorated with '@gui.preprocess', this function:
      * returns the *base* stream used by all plots whenever the relative check-box is on
      * must return either a Trace or Stream object

    - In this implementation THIS FUNCTION DOES MODIFY `segment.stream()` IN-PLACE: from within
      `main`, further calls to `segment.stream()` will return the stream returned by this function.
      However, In any case, you can use `segment.stream().copy()` before this call to keep the
      old "raw" stream

    :return: a Trace object.
    """
    # define some parameters:
    evt = segment.event
    conf = config['preprocess']
    freq_max = conf['bandpass_freq_max']
    if is_accelerometer(segment):
        # accelerometer
        freq_min = mag2freq(evt.magnitude)
    else:
        # velocimeter
        freq_min = conf['velocimeter_freq_min']

    # note: bandpass here below copied the trace! important!
    trace = bandpass(trace, freq_min=freq_min, freq_max=freq_max,
                     max_nyquist_ratio=conf['bandpass_max_nyquist_ratio'],
                     corners=conf['bandpass_corners'], copy=False)
    trace.remove_response(inventory=inventory, output=conf['remove_response_output'],
                          water_level=conf['remove_response_water_level'])
    return trace


def mag2freq(magnitude):
    if magnitude <= 4.:
        freq_min = 0.5
    elif magnitude <= 5.0:
        freq_min = 0.4
    elif magnitude <= 6.0:
        freq_min = 0.2
    elif magnitude <= 6.5:
        freq_min = 0.1
    else:
        freq_min = 0.05
    return freq_min


def _sn_spectra(segment, config):
    """
    Computes the signal and noise spectra, as dict of strings mapped to tuples (x0, dx, y).
    Does not modify the segment's stream or traces in-place

    -Being decorated with '@gui.sideplot' or '@gui.customplot', this function must return
     a numeric sequence y taken at successive equally spaced points in any of these forms:
        - a Trace object
        - a Stream object
        - the tuple (x0, dx, y) or (x0, dx, y, label), where
            - x0 (numeric, `datetime` or `UTCDateTime`) is the abscissa of the first point
            - dx (numeric or `timedelta`) is the sampling period
            - y (numpy array or numeric list) are the sequence values
            - label (string, optional) is the sequence name to be displayed on the plot legend.
              (if x0 is numeric and `dx` is a `timedelta` object, then x0 will be converted
              to `UTCDateTime(x0)`; if x0 is a `datetime` or `UTCDateTime` object and `dx` is
              numeric, then `dx` will be converted to `timedelta(seconds=dx)`)
        - a dict of any of the above types, where the keys (string) will denote each sequence
          name to be displayed on the plot legend.

    :return: a dict with two keys, 'Signal' and 'Noise', mapped respectively to the tuples
    (f0, df, frequencies)

    :raise: an Exception if `segment.stream()` is empty or has more than one trace (possible
    gaps/overlaps)
    """
    arrival_time = UTCDateTime(segment.arrival_time) + config['sn_windows']['arrival_time_shift']
    strace, ntrace = sn_split(segment.stream()[0],  # assumes stream has only one trace
                              arrival_time, config['sn_windows']['signal_window'])
    x0_sig, df_sig, sig = _spectrum(strace, config)
    x0_noi, df_noi, noi = _spectrum(ntrace, config)
    return {'Signal': (x0_sig, df_sig, sig), 'Noise': (x0_noi, df_noi, noi)}


def _spectrum(trace, config):
    '''Calculate the spectrum of a trace. Returns the tuple (0, df, values), where
    values depends on the config dict parameters.
    Does not modify the trace in-place
    '''
    taper_max_percentage = config['sn_spectra']['taper']['max_percentage']
    taper_type = config['sn_spectra']['taper']['type']
    if config['sn_spectra']['type'] == 'pow':
        func = powspec  # copies the trace if needed
    elif config['sn_spectra']['type'] == 'amp':
        func = ampspec  # copies the trace if needed
    else:
        # raise TypeError so that if called from within main, the iteration stops
        raise TypeError("config['sn_spectra']['type'] expects either 'pow' or 'amp'")

    df_, spec_ = func(trace,
                      taper_max_percentage=taper_max_percentage, taper_type=taper_type)

    # if you want to implement your own smoothing, change the lines below before 'return'
    # and implement your own config variables, if any
    smoothing_wlen_ratio = config['sn_spectra']['smoothing_wlen_ratio']
    if smoothing_wlen_ratio > 0:
        spec_ = triangsmooth(spec_, winlen_ratio=smoothing_wlen_ratio)

    return (0, df_, spec_)


def gmpe_reso_14(mag, dist, mode='pga', vs30=800, sof='sofN'):
    '''Gmpe Bindi et al 2014 Rjb

    :param mag: float , magnitude
    :param dist: float, distance in km
    :param mode: string, optional: either 'pga' (the default) or 'pgv'
    :param vs30: float, optional: the vs30 (defaults to 800)
    :param sof: string, optional: the style of faulting, either 'sofN' (the default), 'sofR', sofS'

    :return: The float representing the gmpe's pgv or pga (depending on `mode`),
        in m/sec or m/sec^2, respectively
    '''
    if mode == 'pgv':
        
        #     imt             e1             c1            c2             h            c3             b1             b2            b3          gamma           sofN           sofR           sofS           tau           phi        phis2s         sigma
        #     pgv    2.264810000   -1.224080000   0.202085000   5.061240000   0.000000000    0.162802000   -0.092632400   0.044030100   -0.529443000   -0.009476750    0.040057400   -0.030580500   0.156062000   0.277714000   0.120398000   0.318560000

        e1, c1, c2, h, c3, b1, b2, b3, sA = (2.264810000,
                                             -1.224080000,
                                             0.202085000,
                                             5.061240000,
                                             0.0,
                                             0.162802000,
                                             -0.092632400,
                                             0.044030100,
                                             -0.529443000)
        if sof == 'sofR':
            sof_val = 0.040057400
        elif sof == 'sofS':
            sof_val = -0.030580500
        else:
            sof_val = -0.009476750
        
    else:

        #     imt             e1             c1            c2             h            c3             b1             b2            b3          gamma           sofN           sofR           sofS           tau           phi        phis2s         sigma
        #     pga    3.328190000   -1.239800000   0.217320000   5.264860000   0.001186240   -0.085504500   -0.092563900   0.000000000   -0.301899000   -0.039769500    0.077525300   -0.037755800   0.149977000   0.282398000   0.165611000   0.319753000

        e1, c1, c2, h, c3, b1, b2, b3, sA = (3.328190000,
                                             -1.239800000,
                                             0.217320000,
                                             5.264860000,
                                             0.001186240,
                                             -0.085504500,
                                             -0.092563900,
                                             0.000000000,
                                             -0.301899000)
        if sof == 'sofR':
            sof_val = 0.077525300
        elif sof == 'sofS':
            sof_val = -0.037755800
        else:
            sof_val = -0.039769500

    Mh = 6.75
    Mref = 5.5
    VREF = 800

    exprM1 = mag - Mh
    exprD1 = mag - Mref

    exprM3 = b1 * exprM1 + b2 * exprM1 ** 2
    exprM4 = b3 * exprM1
    valueFM = exprM3 if mag <= Mh else exprM4  # valueFM <- ifelse(MAG<=.Mh, .exprM3, .exprM4)

    Rref = 1.0
    exprD2 = c1 + c2 * exprD1  # [c1 + c2(M-Mref)]
    exprD3 = (dist ** 2 + h ** 2)  # [Rjb^2 + h^2]
    exprD4 = math.sqrt(exprD3)  # [sqrt[Rjb^2 + h^2]]
    exprD5 = exprD4 / Rref      # [sqrt[Rjb^2 + h^2]/Rref]
    exprD6 = math.log10(exprD5)   # LN[sqrt[Rjb^2 + h^2]/Rref]
    exprD8 = exprD4 - Rref  # [sqrt[Rjb^2 + h^2] - Rref]
    valueFD = exprD2 * exprD6 - c3 * exprD8
    
    valueFS = sA * math.log10(vs30 / VREF)

    ## Value ##
    return 10 ** (e1 + valueFD + valueFM + valueFS + sof_val) / 100.0  # returns m/sec or m/sec2


def psd_values(periods, raw_trace, inventory):
    periods = np.asarray(periods)
    ppsd_ = psd(raw_trace, inventory)
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



# GUI RELATED FUNCTIONS (calling already implemented functions above)

@gui.preprocess
def bandpass_remresp(segment, config):
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    trace = stream[0]  # work with the (surely) one trace now
    return __bandpass_remresp(segment, config, trace, segment.inventory())


@gui.customplot
def cumsumsq_normalized(segment, config):
    '''Computes the cumulative of the squares of the segment's trace in the form of a Plot object.
    DOES modify the segment's stream or traces in-place. Normalizes the returned trace values in [0,1]

    -Being decorated with '@gui.sideplot' or '@gui.customplot', this function must return
     a numeric sequence y taken at successive equally spaced points in any of these forms:
        - a Trace object
        - a Stream object
        - the tuple (x0, dx, y) or (x0, dx, y, label), where
            - x0 (numeric, `datetime` or `UTCDateTime`) is the abscissa of the first point
            - dx (numeric or `timedelta`) is the sampling period
            - y (numpy array or numeric list) are the sequence values
            - label (string, optional) is the sequence name to be displayed on the plot legend.
              (if x0 is numeric and `dx` is a `timedelta` object, then x0 will be converted
              to `UTCDateTime(x0)`; if x0 is a `datetime` or `UTCDateTime` object and `dx` is
              numeric, then `dx` will be converted to `timedelta(seconds=dx)`)
        - a dict of any of the above types, where the keys (string) will denote each sequence
          name to be displayed on the plot legend.

    :return: an obspy.Trace

    :raise: an Exception if `segment.stream()` is empty or has more than one trace (possible
    gaps/overlaps)
    '''
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    return cumsumsq(stream[0], normalize=True)


@gui.sideplot
def sn_spectra(segment, config):
    """
    Computes the signal and noise spectra, as dict of strings mapped to tuples (x0, dx, y).
    Does not modify the segment's stream or traces in-place

    -Being decorated with '@gui.sideplot' or '@gui.customplot', this function must return
     a numeric sequence y taken at successive equally spaced points in any of these forms:
        - a Trace object
        - a Stream object
        - the tuple (x0, dx, y) or (x0, dx, y, label), where
            - x0 (numeric, `datetime` or `UTCDateTime`) is the abscissa of the first point
            - dx (numeric or `timedelta`) is the sampling period
            - y (numpy array or numeric list) are the sequence values
            - label (string, optional) is the sequence name to be displayed on the plot legend.
              (if x0 is numeric and `dx` is a `timedelta` object, then x0 will be converted
              to `UTCDateTime(x0)`; if x0 is a `datetime` or `UTCDateTime` object and `dx` is
              numeric, then `dx` will be converted to `timedelta(seconds=dx)`)
        - a dict of any of the above types, where the keys (string) will denote each sequence
          name to be displayed on the plot legend.

    :return: a dict with two keys, 'Signal' and 'Noise', mapped respectively to the tuples
    (f0, df, frequencies)

    :raise: an Exception if `segment.stream()` is empty or has more than one trace (possible
    gaps/overlaps)
    """
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    return _sn_spectra(segment, config)
