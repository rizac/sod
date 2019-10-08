'''
=============================================================
Stream2segment: Processing+Visualization python file template
=============================================================

This is a template python module for processing downloaded waveform segments and defining the
segment plots to be visualized in the web GUI (Graphical user interface).

This file can be edited and passed to the program commands `s2s v` (visualize) and `s2s p` (process)
as -p or --pyfile option, together with an associated configuration .yaml file (-c option):
```
    s2s show -p [thisfilepath] -c [configfilepath] ...
    s2s process -p [thisfilepath] -c [configfilepath] ...
```
This module needs to implement one or more functions which will be described here (for full details,
look at their doc-string). All these functions must have the same signature:
```
    def myfunction(segment, config):
```
where `segment` is the python object representing a waveform data segment to be processed
and `config` is the python dictionary representing the given configuration .yaml file.

Processing
==========

When invoked via `s2s process ...`, the program will search for a function called "main", e.g.:
```
def main(segment, config)
```
the program will iterate over each selected segment (according to 'segment_select' parameter
in the config) and execute the function, writing its output to the given .csv file

Visualization (web GUI)
=======================

When invoked via `s2s show ...`, the program will search for all functions decorated with
"@gui.preprocess", "@gui.sideplot" or "@gui.customplot".
The function decorated with "@gui.preprocess", e.g.:
```
@gui.preprocess
def applybandpass(segment, config)
```
will be associated to a check-box in the GUI. By clicking the check-box,
all plot functions (i.e., all other functions decorated with either '@sideplot' or '@customplot')
are re-executed with the only difference that `segment.stream()`
will return the pre-processed stream, instead of the "raw" unprocessed stream. Thus, this
function must return a Stream or Trace object.
The function decorated with "@gui.sideplot", e.g.:
```
@gui.sideplot
def sn_spectra(segment, config)
```
will be associated to (i.e., its output will be displayed in) the right side plot,
next to the raw / pre-processed segment stream plot.
Finally, the functions decorated with "@gui.customplot", e.g.:
```
@gui.customplot
def cumulative(segment, config)
@gui.customplot
def first_derivative(segment, config)
...
```
will be associated to the bottom plot, below the raw / pre-processed segment stream plot, and can
be selected (one at a time) from the GUI with a radio-button.
All plot functions should return objects of certain types (more details in their doc-strings
in this module).

Important notes
===============

1) This module is designed to force the DRY (don't repeat yourself) principle, thus if a portion
of code implemented in "main" should be visualized for inspection, it should be moved, not copied,
to a separated function (decorated with '@gui.customplot') and called from within "main"

2) All functions here can safely raise Exceptions, as all exceptions will be caught by the caller:
- displaying the error message on the plot if the function is called for visualization,
- printing it to a log file, if teh function is called for processing into .csv
  (More details on this in the "main" function doc-string).
Thus do not issue print statement in any function because, to put it short,
it's useless (and if you are used to do it extensively for debugging, consider changing this habit
and use a logger): if any information should be given, simply raise a base exception, e.g.:
`raise Exception("segment sample rate too low")`.

Functions arguments
===================

As said, all functions needed or implemented for processing and visualization must have the same
signature:
```
    def myfunction(segment, config):
```
In details, the two arguments passed to those functions are:

segment (object)
~~~~~~~~~~~~~~~~

Technically it's like an 'SqlAlchemy` ORM instance but for the user it is enough to
consider and treat it as a normal python object. It has special methods and several
attributes returning python "scalars" (float, int, str, bool, datetime, bytes).
Each attribute can be considered as segment metadata: it reflects a segment column
(or an associated database table via a foreign key) and returns the relative value.

segment methods:
----------------

* segment.stream(): the `obspy.Stream` object representing the waveform data
  associated to the segment. Please remember that many obspy functions modify the
  stream in-place:
  ```
      s = segment.stream()
      s_rem_resp = s.remove_response(segment.inventory())
      segment.stream() is s  # False!!!
      segment.stream() is s_rem_resp  # True!!!
  ```
  When visualizing plots, where efficiency is less important, each function is executed on a
  copy of segment.stream(). However, from within the `main` function, the user has to handle when
  to copy the segment's stream or not. For info see:
  https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.copy.html

* segment.inventory(): the `obspy.core.inventory.inventory.Inventory`. This object is useful e.g.,
  for removing the instrumental response from `segment.stream()`

* segment.sn_windows(): returns the signal and noise time windows:
  (s_start, s_end), (n_start, n_end)
  where all elements are `UTCDateTime`s. The windows are computed according to
  the settings of the associated yaml configuration file: `config['sn_windows']`). Example usage:
  `
  sig_wdw, noise_wdw = segment.sn_windows()
  stream_noise = segment.stream().copy().trim(*noise_wdw, ...)
  stream_signal = segment.stream().copy().trim(*sig_wdw, ...)
  `
  If segment's stream has more than one trace, the method raises.

* segment.other_orientations(): returns a list of segment objects representing the same recorded
  event on other channel's orientations. E.g., if `segment` refers to an event E recorded by a
  station channel with code 'HHZ', this method returns the segments recorded on 'HHE' and
  'HHN' (relative to the same event on the same station and location codes).

* segment.del_classes(*ids_or_labels): Deletes the given classes of the segment. The argument is
  a comma-separated list of class labels (string) or class ids (int). As classes are given in the
  config as a dict of label:description values, usually string labels are passed here.
  E.g.: `segment.del_classes('class1')`, `segment.del_classes('class1', 'class2', 'class3')`

* segment.set_classes(*ids_or_labels, annotator=None): Sets the given classes on the segment,
  deleting all already assigned classes, if any. `ids_or_labels` is a comma-separated list of class
  labels (string) or class ids (int). As classes are given in the config as a dict of
  label:description values, usually string labels are passed here. `annotator` is a string name
  which, if given (not None) denotes that the class labelling is a human hand-labelled class
  assignment (vs. a statistical classifier class assignment).
  E.g.: `segment.set_classes('class1')`, `segment.set_classes('class1', 'class2', annotator='Jim')`

* segment.add_classes(*ids_or_labels, annotator=None): Same as `segment.set_classes` but already
  assigned classes will neither be deleted first, nor added again if already assigned

* segment.dbsession(): WARNING: this is for advanced users experienced with Sql-Alchemy library:
  returns the database session for IO operations with the database


segment attributes:
-------------------

========================================= ================================================
attribute                                 python type and description (if any)
========================================= ================================================
segment.id                                int: segment (unique) db id
segment.event_distance_deg                float: distance between the segment's station and
\                                         the event, in degrees
segment.event_distance_km                 float: distance between the segment's station and
\                                         the event, in km, assuming a perfectly spherical earth
\                                         with a radius of 6371 km
segment.start_time                        datetime.datetime: the waveform data start time
segment.arrival_time                      datetime.datetime
segment.end_time                          datetime.datetime: the waveform data end time
segment.request_start                     datetime.datetime: the requested start time of the data
segment.request_end                       datetime.datetime: the requested end time of the data
segment.duration_sec                      float: the waveform data duration, in seconds
segment.missing_data_sec                  float: the number of seconds of missing data, with respect
\                                         to the request time window. E.g. if we requested 5
\                                         minutes of data and we got 4 minutes, then
\                                         missing_data_sec=60; if we got 6 minutes, then
\                                         missing_data_sec=-60. This attribute is particularly
\                                         useful in the config to select only well formed data and
\                                         speed up the processing, e.g.: missing_data_sec: '< 120'
segment.missing_data_ratio                float: the portion of missing data, with respect
\                                         to the request time window. E.g. if we requested 5
\                                         minutes of data and we got 4 minutes, then
\                                         missing_data_ratio=0.2 (20%); if we got 6 minutes, then
\                                         missing_data_ratio=-0.2. This attribute is particularly
\                                         useful in the config to select only well formed data and
\                                         speed up the processing, e.g.: missing_data_ratio: '< 0.5'
segment.has_data                          boolean: tells if the segment has data saved (at least
\                                         one byte of data). This attribute useful in the config to
\                                         select only well formed data and speed up the processing,
\                                         e.g. has_data: 'true'.
segment.sample_rate                       float: the waveform data sample rate.
\                                         It might differ from the segment channel's sample_rate
segment.download_code                     int: the download code (for experienced users). As for
\                                         any HTTP status code,
\                                         values between 200 and 399 denote a successful download
\                                         (this does not tell anything about the segment's data,
\                                         which might be empty anyway. See 'segment.has_data'.
\                                         Conversely, a download error assures no data has been
\                                         saved), whereas
\                                         values >=400 and < 500 denote client errors and
\                                         values >=500 server errors.
\                                         Moreover,
\                                         -1 indicates a general download error - e.g. no Internet
\                                         connection,
\                                         -2 a successful download with corrupted waveform data,
\                                         -200 a successful download where some waveform data chunks
\                                         (miniSeed records) have been discarded because completely
\                                         outside the requested time span,
\                                         -204 a successful download where no data has been saved
\                                         because all chunks were completely outside the requested
\                                         time span, and finally:
\                                         None denotes a successful download where no data has been
\                                         saved because the given segment wasn't found in the
\                                         server response (note: this latter case is NOT the case
\                                         when the server returns no data with an appropriate
\                                         'No Content' message with download_code=204)
segment.maxgap_numsamples                 float: the maximum gap found in the waveform data, in
\                                         in number of points.
\                                         If the value is positive, the max is a gap. If negative,
\                                         it's an overlap. If zero, no gaps/overlaps were found.
\                                         This attribute is particularly useful in the config to
\                                         select only well formed data and speed up the processing,
\                                         e.g.: maxgap_numsamples: '[-0.5, 0.5]'.
\                                         This number is a float because it is the ratio between
\                                         the waveform data's max gap/overlap and its sampling
\                                         period (both in seconds). Thus, non-zero float values
\                                         in (-1, 1) are difficult to interpret: a rule of thumb
\                                         is to consider a segment with gaps/overlaps when this
\                                         attribute's absolute value exceeds 0.5. The user can
\                                         always perform a check in the processing for
\                                         safety, e.g., via `len(segment.stream())` or
\                                         `segment.stream().get_gaps()`)
segment.data_seed_id                      str: the seed identifier in the typical format
\                                         [Network.Station.Location.Channel] stored in the
\                                         segment's data. It might be null if the data is empty
\                                         or null (e.g., because of a download error).
\                                         See also 'segment.seed_id'
segment.seed_id                           str: the seed identifier in the typical format
\                                         [Network.Station.Location.Channel]: it is the same as
\                                         'segment.data_seed_id' if the latter is not null,
\                                         otherwise it is fetched from the segment's metadata
\                                         (this operation might be more time consuming)
segment.has_class                         boolean: tells if the segment has (at least one) class
\                                         assigned
segment.data                              bytes: the waveform (raw) data. You don't generally need
\                                         to access this attribute which is also time-consuming
\                                         to fetch. Used by `segment.stream()`
----------------------------------------- ------------------------------------------------
segment.event                             object (attributes below)
segment.event.id                          int
segment.event.event_id                    str: the id returned by the web service
segment.event.time                        datetime.datetime
segment.event.latitude                    float
segment.event.longitude                   float
segment.event.depth_km                    float
segment.event.author                      str
segment.event.catalog                     str
segment.event.contributor                 str
segment.event.contributor_id              str
segment.event.mag_type                    str
segment.event.magnitude                   float
segment.event.mag_author                  str
segment.event.event_location_name         str
----------------------------------------- ------------------------------------------------
segment.channel                           object (attributes below)
segment.channel.id                        int
segment.channel.location                  str
segment.channel.channel                   str
segment.channel.depth                     float
segment.channel.azimuth                   float
segment.channel.dip                       float
segment.channel.sensor_description        str
segment.channel.scale                     float
segment.channel.scale_freq                float
segment.channel.scale_units               str
segment.channel.sample_rate               float
segment.channel.band_code                 str: the first letter of channel.channel
segment.channel.instrument_code           str: the second letter of channel.channel
segment.channel.orientation_code          str: the third letter of channel.channel
segment.channel.station                   object: same as segment.station (see below)
----------------------------------------- ------------------------------------------------
segment.station                           object (attributes below)
segment.station.id                        int
segment.station.network                   str
segment.station.station                   str
segment.station.latitude                  float
segment.station.longitude                 float
segment.station.elevation                 float
segment.station.site_name                 str
segment.station.start_time                datetime.datetime
segment.station.end_time                  datetime.datetime
segment.station.inventory_xml             bytes. The station inventory (raw) data. You don't
\                                         generally need to access this attribute which is also
\                                         time-consuming to fetch. Used by `segment.inventory()`
segment.station.has_inventory             boolean: tells if the segment's station inventory has
\                                         data saved (at least one byte of data).
\                                         This attribute useful in the config to select only
\                                         segments with inventory downloaded and speed up the
\                                         processing,
\                                         e.g. has_inventory: 'true'.
segment.station.datacenter                object (same as segment.datacenter, see below)
----------------------------------------- ------------------------------------------------
segment.datacenter                        object (attributes below)
segment.datacenter.id                     int
segment.datacenter.station_url            str
segment.datacenter.dataselect_url         str
segment.datacenter.organization_name      str
----------------------------------------- ------------------------------------------------
segment.download                          object (attributes below): the download execution
segment.download.id                       int
segment.download.run_time                 datetime.datetime
segment.download.log                      str: The log text of the segment's download execution.
\                                         You don't generally need to access this
\                                         attribute which is also time-consuming to fetch.
\                                         Useful for advanced debugging / inspection
segment.download.warnings                 int
segment.download.errors                   int
segment.download.config                   str
segment.download.program_version          str
========================================= ================================================

config (dict)
~~~~~~~~~~~~~

This is the dictionary representing the chosen .yaml config file (usually, via command line).
By design, we strongly encourage to decouple code and configuration, so that you can easily
and safely experiment different configurations on the same code, if needed.
The config default file is documented with all necessary information, and you can put therein
whatever you want to be accessible as a python dict key, e.g. `config['mypropertyname']`
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
        raise Exception("%d traces (probably gaps/overlaps)" % len(stream))


def main(segment, config):
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


    # add gain (2):
#     stage_gains = []
#     cha_obj = get_cha_obj(segment)
#     for stage in cha_obj.response.response_stages:
#         stage_gains.append(stage.stage_gain)
#     for gain_factor in config['stage_gain_factors']:
#         cha_obj = get_cha_obj(segment)
#         for stage, original_gain in zip(cha_obj.response.response_stages, stage_gains):
#             stage.stage_gain = original_gain * gain_factor
#         data.append(_main(segment, config, trace.copy(), segment.inventory()))
#         data[-1]['outlier'] = 1
#         data[-1]['modified'] = "RESPGAIN:X%f" % gain_factor
#     for stage, original_gain in zip(cha_obj.response.response_stages, stage_gains):
#         stage.stage_gain = original_gain
    
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
    Main processing function for generating output in a .csv file
    See `return` below for a detailed explanation of what this function should return after the
    processing is completed

    This function is called by executing the command:
    ```
        >>> stream2segment -p $PYFILE -c $CONFIG $OUTPUT
    ```
    where:
      - $PYFILE is the path of this file,
      - $CONFIG is a path to the .yaml configuration file (if this file was auto generated,
        it should be a file named $FILE.yaml)
      - $OUTPUT is the csv file where data (one row per segment) will to be saved

    For info about possible functions to use, please have a look at `stream2segment.analysis.mseeds`
    and obviously at `obpsy <https://docs.obspy.org/packages/index.html>`_, in particular:

    *  `obspy.core.Stream <https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.html#obspy.core.stream.Stream>_`
    *  `obspy.core.Trace <https://docs.obspy.org/packages/autogen/obspy.core.trace.Trace.html#obspy.core.trace.Trace>_`

    :param: segment (ptyhon object): An object representing a waveform data to be processed,
    reflecting the relative database table row. See module docstring above for a detailed list
    of attributes and methods

    :param: config (python dict): a dictionary reflecting what has been implemented in $CONFIG.
    You can write there whatever you want (in yaml format, e.g. "propertyname: 6.7" ) and it
    will be accessible as usual via `config['propertyname']`

    :return: an iterable (list, tuple, numpy array, dict...) of values. The returned iterable
    will be written as a row of the resulting csv file. If dict, the keys of the dict
    will populate the first row header of the resulting csv file, otherwise the csv file
    will have no header. Please be consistent: always return the same type of iterable for
    all segments; if dict, always return the same keys for all dicts; if list, always
    return the same length, etcetera.
    If you want to preserve the order of the dict keys as inserted in the code, use `OrderedDict`
    instead of `dict` or `{}`.
    Please note that the first column of the resulting csv will be *always* the segment id
    (an integer stored in the database uniquely identifying the segment). Thus the first value
    returned by the iterable of `main` will be in the csv file second column, the second in the
    third, and so on.
    If this function (or module, when imported) or any of the functions called raise any of the
    following:
    `TypeError`, `SyntaxError`, `NameError`, `ImportError`, `AttributeError`
    then the whole process will **stop**, because those exceptions are most likely caused
    by code errors which might affect all segments and the user can fix them without waiting
    for all segments to be processed.
    Otherwise, the function can **raise** any *other* Exception, or **return** None.
    In both cases, the iteration will not stop but will go on processing the following segment.
    None will silently ignore the segment, otherwise
    the exception message (with the segment id) will be written to a .log file in the same folder
    than the output csv file.
    Pay attention when setting complex objects (e.g., everything neither string nor numeric) as
    elements of the returned iterable: the values will be most likely converted to string according
    to python `__str__` function and might be out of control for the user.
    Thus, it is suggested to convert everything to string or number. E.g., for obspy's
    `UTCDateTime`s you could return either `float(utcdatetime)` (numeric) or
    `utcdatetime.isoformat()` (string)
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
#     snr1_ = snr(normal_spe, noise_spe, signals_form=config['sn_spectra']['type'],
#                fmin=fcmin, fmax=1, delta_signal=normal_df, delta_noise=noise_df)
#     snr2_ = snr(normal_spe, noise_spe, signals_form=config['sn_spectra']['type'],
#                fmin=1, fmax=10, delta_signal=normal_df, delta_noise=noise_df)
#     snr3_ = snr(normal_spe, noise_spe, signals_form=config['sn_spectra']['type'],
#                fmin=10, fmax=fcmax, delta_signal=normal_df, delta_noise=noise_df)
    low_snr = False
    if snr_ < config['snr_threshold']:
        low_snr = True

    # calculate cumulative

    cum_labels = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    cum_trace = cumsumsq(trace, normalize=True, copy=True)  # prevent original trace from being modified
    cum_times = timeswhere(cum_trace, *cum_labels)

    # calculate PGA and times of occurrence (t_PGA):
    t_PGA, PGA = maxabs(trace, cum_times[1], cum_times[-2])  # note: you can also provide tstart tend for slicing
    trace_int = trace.copy().integrate()
    t_PGV, PGV = maxabs(trace_int, cum_times[1], cum_times[-2])
#     meanoff = meanslice(trace_int, 100, cum_times[-1], trace_int.stats.endtime)

    # calculates amplitudes at the frequency bins given in the config file:
    required_freqs = config['freqs_interp']
    ampspec_freqs = normal_f0 + normal_df * np.arange(len(normal_spe))
    required_amplitudes = np.interp(np.log10(required_freqs),
                                    np.log10(ampspec_freqs),
                                    normal_spe) / segment.sample_rate

    # compute synthetic WA.
    # IMPORTANT: modifies the segment trace in-place!
#     trace_wa = synth_wa(segment, config)
#     t_WA, maxWA = maxabs(trace_wa)

    # write stuff to csv:
    ret = OrderedDict()

    # pga_pre, pgv_pre, pga_obs_, pgv_obs, f0.5_obs f1_obs f2_obs f5_obs f10_obs (in Hertz)
    # f0.5_pre f1_pre f2_pre f5_pre f10_pre (in Hertz, for the moment empty)
    # Mag, dist, snr, saturated

    distance = segment.event_distance_km
    ret['amplitude_ratio'] = amp_ratio
    ret['saturated'] = saturated
    ret['snr'] = snr_
    ret['low_snr'] = low_snr
    ret['magnitude'] = magnitude
    ret['distance_km'] = distance
    ret['event_time'] = segment.event.time
#     ret['snr1'] = snr1_
#     ret['snr2'] = snr2_
#     ret['snr3'] = snr3_
#     for cum_lbl, cum_t in zip(cum_labels[slice(1,8,3)], cum_times[slice(1,8,3)]):
#         ret['cum_t%f' % cum_lbl] = float(cum_t)  # convert cum_times to float for saving

#     ret['dist_deg'] = segment.event_distance_deg        # dist
#     ret['dist_km'] = d2km(segment.event_distance_deg)  # dist_km
#     ret['t_PGA'] = t_PGA                  # peak info
    ret['pga_observed'] = PGA
    ret['pga_predicted'] = gmpe_reso_14(magnitude, distance, mode='pga')
#     ret['t_PGV'] = t_PGV                  # peak info
    ret['pgv_observed'] = PGV
    ret['pgv_predicted'] = gmpe_reso_14(magnitude, distance, mode='pgv')

    # ret['PGA_diff'] = PGA - ret['PGA_predicted']
    # ret['PGV_diff'] = PGV - ret['PGV_predicted']

    periods = config['psd_periods']
    psdvalues = psd_values(segment, periods, inventory)
    for f, a in zip(periods, psdvalues):
        ret['psd@%ssec' % str(f)] = float(a)
#     ret['t_WA'] = t_WA
#     ret['maxWA'] = maxWA
#     ret['channel'] = segment.channel.channel
#     ret['channel_component'] = segment.channel.channel[-1]
#     ret['ev_id'] = segment.event.id           # event metadata
#     ret['ev_lat'] = segment.event.latitude
#     ret['ev_lon'] = segment.event.longitude
#     ret['ev_dep'] = segment.event.depth_km
#     ret['ev_mag'] = segment.event.magnitude
#     ret['ev_mty'] = segment.event.mag_type
#     ret['st_id'] = segment.station.id         # station metadata
#     ret['st_name'] = segment.station.station
#     ret['st_net'] = segment.station.network
#     ret['st_lat'] = segment.station.latitude
#     ret['st_lon'] = segment.station.longitude
#     ret['st_ele'] = segment.station.elevation
#     ret['offset'] = np.abs(meanoff/PGV)
    for f, a in zip(required_freqs, required_amplitudes):
        ret['%s@%shz' % (spectrum_type, str(f))] = float(a)
#     ret['channel_location']=segment.channel.location
#     ret['channel_depth']=segment.channel.depth
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
