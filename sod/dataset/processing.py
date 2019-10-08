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
    timeswhere, fft, maxabs, utcdatetime, ampspec, powspec, timeof
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
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    trace = stream[0]  # work with the (surely) one trace now

    data = []
    data.append(_main(segment, config, trace.copy(), segment.inventory()))

    # add gain (1)
    for gain_factor in config['stage_gain_factors']:
        data.append(_main(segment, config, trace.copy(), segment.inventory(), gain_factor))
        data[-1]['outlier'] = 1
        data[-1]['modified'] = "STAGEGAIN:X%s" % str(gain_factor)

    # acceleromters/velocimeters:
    if segment.station.id in config['station_ids_both_accel_veloc']:
        # reload inventory (takes more time, but we won't modify cached version):
        inventory = get_inventory(segment.station)
        cha_obj = get_cha_obj(segment, inventory)
        resp_tmp = cha_obj.response
        for other_cha in get_other_chan_objs(segment, inventory):
            cha_obj.response = other_cha.response
            data.append(_main(segment, config, trace.copy(), inventory))
            data[-1]['outlier'] = 1
            data[-1]['modified'] = "CHARESP:%s" % other_cha.code
        cha_obj.response = resp_tmp

    if segment.station.id in config['station_ids_with_wrong_local_inventory']:
        channels_ = config['station_ids_with_wrong_local_inventory'][segment.station.id]
        filename = channels_.get(segment.data_seed_id, None)
        if filename is None:
            raise ValueError('%s not found in wrong inventories dict' % segment.data_seed_id)
        if filename is not None:
            inventories_dir = config['inventories_dir']
            wrong_inventory = read_inventory(os.path.join(os.getcwd(), inventories_dir,
                                                          filename))
            data.append(_main(segment, config, trace.copy(), wrong_inventory))
            data[-1]['outlier'] = 1
            data[-1]['modified'] = "INVFILE:%s" % filename

    return pd.DataFrame(data)


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


def _main(segment, config, trace, inventory, gain_factor=None):
    """
    called by main with supplied inventory
    """
    # discard saturated signals (according to the threshold set in the config file):
    saturated = False
    amp_ratio = ampratio(trace)
    if amp_ratio >= config['amp_ratio_threshold']:
        saturated = True

    # bandpass the trace, according to the event magnitude.
    # WARNING: this modifies the segment.stream() permanently!
    # If you want to preserve the original stream, store trace.copy()
    trace = _bandpass_remresp(segment, config, trace, inventory)

    if gain_factor is not None:
        trace.data = trace.data * float(gain_factor)  # might be int, let's be sure...

    spectra = _sn_spectra(segment, config, trace)
    normal_f0, normal_df, normal_spe = spectra['Signal']
    noise_f0, noise_df, noise_spe = spectra['Noise']
    magnitude = segment.event.magnitude
    fcmin = mag2freq(magnitude)
    fcmax = config['preprocess']['bandpass_freq_max']  # used in bandpass_remresp
    spectrum_type = config['sn_spectra']['type']
    snr_ = snr(normal_spe, noise_spe, signals_form=spectrum_type,
               fmin=fcmin, fmax=fcmax, delta_signal=normal_df, delta_noise=noise_df)

    low_snr = False
    if snr_ < config['snr_threshold']:
        low_snr = True

    # calculate cumulative

    cum_labels = [0.05, 0.95]
    # copy=True: prevent original trace from being modified:
    cum_trace = cumsumsq(trace, normalize=True, copy=True)
    cum_times = timeswhere(cum_trace, *cum_labels)

    # calculate PGA and times of occurrence (t_PGA):
    t_PGA, PGA = maxabs(trace, cum_times[0], cum_times[-1])
    trace_int = trace.copy().integrate()
    t_PGV, PGV = maxabs(trace_int, cum_times[0], cum_times[-1])
#     meanoff = meanslice(trace_int, 100, cum_times[-1], trace_int.stats.endtime)

    # calculates amplitudes at the frequency bins given in the config file:
    required_freqs = config['freqs_interp']
    ampspec_freqs = normal_f0 + normal_df * np.arange(len(normal_spe))
    required_amplitudes = np.interp(np.log10(required_freqs),
                                    np.log10(ampspec_freqs),
                                    normal_spe) / segment.sample_rate

    # write stuff to csv:
    ret = OrderedDict()

    distance = segment.event_distance_km

    ret['amplitude_ratio'] = amp_ratio
    ret['saturated'] = saturated
    ret['snr'] = snr_
    ret['low_snr'] = low_snr
    ret['magnitude'] = magnitude
    ret['distance_km'] = distance
    ret['event_time'] = segment.event.time
    ret['pga_observed'] = PGA
    ret['pga_predicted'] = gmpe_reso_14(magnitude, distance, mode='pga')
    ret['pgv_observed'] = PGV
    ret['pgv_predicted'] = gmpe_reso_14(magnitude, distance, mode='pgv')

    periods = config['psd_periods']
    psdvalues = psd_values(segment, periods, inventory)
    for f, a in zip(periods, psdvalues):
        ret['psd@%ssec' % str(f)] = float(a)

    for f, a in zip(required_freqs, required_amplitudes):
        ret['%s@%shz' % (spectrum_type, str(f))] = float(a)

    ret['outlier'] = 0
    ret['modified'] = ''

    return ret


def _bandpass_remresp(segment, config, trace, inventory):
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


def _sn_spectra(segment, config, trace):
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
    signal_wdw, noise_wdw = segment.sn_windows(config['sn_windows']['signal_window'],
                                               config['sn_windows']['arrival_time_shift'])
    x0_sig, df_sig, sig = _spectrum(trace, config, *signal_wdw)
    x0_noi, df_noi, noi = _spectrum(trace, config, *noise_wdw)
    return {'Signal': (x0_sig, df_sig, sig), 'Noise': (x0_noi, df_noi, noi)}


def _spectrum(trace, config, starttime=None, endtime=None):
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

    df_, spec_ = func(trace, starttime, endtime,
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


def psd_values(segment, periods, inventory=None):
    periods = np.asarray(periods)
    ppsd_ = psd(segment, inventory)
    val = np.interp(np.log10(periods), np.log10(ppsd_.period_bin_centers), ppsd_.psd_values[0])
    val[periods < ppsd_.period_bin_centers[0]] = np.nan
    val[periods > ppsd_.period_bin_centers[-1]] = np.nan
    return val


def psd(segment, inventory=None):
    if inventory is None:
        inventory = segment.inventory()
    tr = segment.stream()[0]
    dt = (segment.arrival_time - tr.stats.starttime.datetime).total_seconds()
    tr = tr.slice(tr.stats.starttime, UTCDateTime(segment.arrival_time))
    ppsd = PPSD(tr.stats, metadata=inventory, ppsd_length=int(dt))
    ppsd.add(tr)
    return ppsd



# GUI RELATED FUNCTIONS (calling already implemented functions above)

@gui.preprocess
def bandpass_remresp(segment, config):
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    trace = stream[0]  # work with the (surely) one trace now
    return _bandpass_remresp(segment, config, trace, segment.inventory())


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
    return _sn_spectra(segment, config, stream[0])
