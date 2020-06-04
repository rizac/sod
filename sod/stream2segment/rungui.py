'''
This editable template shows how to generate a segment-based parametric table.
When used for visualization, this template implements a pre-processing function that
can be toggled with a checkbox, and several task-specific functions (e.g., cumulative,
synthetic Wood-Anderson) that can to be visualized as custom plots in the GUI

============================================================================================
stream2segment Python file to implement the processing/visualization subroutines: User guide
============================================================================================

This module needs to implement one or more functions which will be described in the sections below.
**All these functions must have the same signature**:
```
    def myfunction(segment, config):
```
where `segment` is the Python object representing a waveform data segment to be processed
and `config` is the Python dictionary representing the given configuration file.

After editing, this file can be invoked from the command line commands `s2s process` and `s2s show`
with the `-p` / `--pyfile` option (type `s2s process --help` or `s2s show --help` for details).
In the first case, see section 'Processing' below, otherwise see section 'Visualization (web GUI)'.


Processing
==========

When processing, the program will search for a function called "main", e.g.:
```
def main(segment, config)
```
the program will iterate over each selected segment (according to 'segment_select' parameter
in the config) and execute the function, writing its output to the given file, if given.
If you do not need to use this module for visualizing stuff, skip the section 'Visualization'
below and go to "Functions implementation".


Visualization (web GUI)
=======================

When visualizing, the program will fetch all segments (according
to 'segment_select' parameter in the config), and open a web page where the user can browse and
visualize each segment one at a time.
The page shows by default on the upper left corner a plot representing the segment trace(s).
The GUI can be customized by providing here functions decorated with
"@gui.preprocess" or "@gui.plot".
Functions decorated this way (Plot functions) can return only special 'plottable' values
(see 'Plot functions' below for details).

Pre-process function
--------------------

The function decorated with "@gui.preprocess", e.g.:
```
@gui.preprocess
def applybandpass(segment, config)
```
will be associated to a check-box in the GUI. By clicking the check-box,
all plots of the page will be re-calculated with the output of this function,
which **must thus return an obspy Stream or Trace object**.

Plot functions
--------------

The function decorated with "@gui.plot", e.g.:
```
@gui.plot
def cumulative(segment, config)
...
```
will be associated to (i.e., its output will be displayed in) the plot below the main plot.
You can also call @gui.plot with arguments, e.g.:
```
@gui.plot(position='r', xaxis={'type': 'log'}, yaxis={'type': 'log'})
def spectra(segment, config)
...
```
The 'position' argument controls where the plot will be placed in the GUI ('b' means bottom,
the default, 'r' means next to the main plot, on its right) and the other two, `xaxis` and
`yaxis`, are dict (defaulting to the empty dict {}) controlling the x and y axis of the plot
(for info, see: https://plot.ly/python/axes/). When not given, axis types will be inferred
from the function's return type (see below) and in most cases defaults to 'date' (i.e.,
date-times on the x values).

Functions decorated with '@gui.plot' must return a numeric sequence y taken at successive
equally spaced points in any of these forms:

- a obspy Trace object

- a obspy Stream object

- the tuple (x0, dx, y) or (x0, dx, y, label), where

    - x0 (numeric, `datetime` or `UTCDateTime`) is the abscissa of the first point.
      For time-series abscissas, UTCDateTime is quite flexible with several input formats.
      For info see: https://docs.obspy.org/packages/autogen/obspy.core.utcdatetime.UTCDateTime.html

    - dx (numeric or `timedelta`) is the sampling period. If x0 has been given as date-time
      or UTCDateTime object and 'dx' is numeric, its unit is in seconds
      (e.g. 45.67 = 45 seconds and 670000 microseconds). If `dx` is a timedelta object and
      x0 has been given as numeric, then x0 will be converted to UtcDateTime(x0).

    - y (numpy array or numeric list) are the sequence values, numeric

    - label (string, optional) is the sequence name to be displayed on the plot legend.

- a dict of any of the above types, where the keys (string) will denote each sequence
  name to be displayed on the plot legend (and will override the 'label' argument, if provided)

Functions implementation
========================

The implementation of the functions is user-dependent. As said, all functions needed for
processing and visualization must have the same signature:
```
    def myfunction(segment, config):
```

any Exception raised will be handled this way:

* if the function is called for visualization, the exception will be caught and its message
  displayed on the plot

* if the function is called for processing, the exception will raise as usual, interrupting
  the routine, with one special case: `ValueError`s will interrupt the currently processed segment
  only (the exception message will be logged) and continue the execution to the next segment.
  This feature can also be triggered programmatically to skip the currently processed segment and
  log the error for later insopection, e.g.:
    `raise ValueError("segment sample rate too low")`
  (thus, do not issue `print` statements for debugging as it's useless, and a bad practice overall)

Conventions and suggestions
---------------------------

Handling exceptions, especially when launching a long processing routine, might be non trivial.
There is a tradeoff to choose: just implement your code and run it (but the whole execution
might stop days later, just shortly before ending), or catch any potential exception
and raise a `ValueError` to continue the execution to the next segment (but you might realise
too late that all segments raised errors and no data has actually been written to file).

If you go for the former (the cleanest one), we suggest to run your routine on a smaller
(and possibly heterogeneous) dataset first (using the configuration file's segment selection):
this not only allows you to handle potential unexpected exceptions, but also to inspect the
output and debug your code.
If you go for the latter, try to inspect the log file (especially at the beginning of
the whole execution) to be ready to stop the run if you see something suspicious, avoiding waste
of time.

In both cases, please spend some time on the configuration file's segment selection: you might find
that your code runs smoothly as expected (and faster) by simply skipping certain segments in
the first place.

This module is designed to encourage the decoupling of code and configuration, so that you can
easily and safely experiment different configurations on the same code of the same Python module.
Avoid having duplicated modules with different hard coded parameters.

Functions arguments
-------------------

config (dict)
~~~~~~~~~~~~~

This is the dictionary representing the chosen configuration file (usually, via command line)
in YAML format (see documentation therein). Any property defined in the configuration file, e.g.:
```
outfile: '/home/mydir/root'
mythreshold: 5.67
```
will be accessible via `config['outfile']`, `config['mythreshold']`

segment (object)
~~~~~~~~~~~~~~~~

Technically it's like an 'SqlAlchemy` ORM instance but for the user it is enough to
consider and treat it as a normal Python object. It features special methods and
several attributes returning Python "scalars" (float, int, str, bool, datetime, bytes).
Each attribute can be considered as segment metadata: it reflects a segment column
(or an associated database table via a foreign key) and returns the relative value.

segment methods:
----------------

* segment.stream(reload=False): the `obspy.Stream` object representing the waveform data
  associated to the segment. Please remember that many obspy functions modify the
  stream in-place:
  ```
      stream_remresp = segment.stream().remove_response(segment.inventory())
      # any call to segment.stream() returns from now on `stream_remresp`
  ```
  For any case where you do not want to modify `segment.stream()`, copy the stream
  (or its traces) first, e.g.:
  ```
      stream_raw = segment.stream()
      stream_remresp = stream_raw.copy().remove_response(segment.inventory())
      # any call to segment.stream() will still return `stream_raw`
  ```
  You can also pass a boolean value (False by default when missing) to `stream` to force
  reloading it from the database (this is less performant as it resets the cached value):
  ```
      stream_remresp = segment.stream().remove_response(segment.inventory())
      stream_reloaded = segment.stream(True)
      # any call to segment.stream() returns from now on `stream_reloaded`
  ```
  (In visualization functions, i.e. those decorated with '@gui', any modification
  to the segment stream will NOT affect the segment's stream in other functions)

  For info see https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.copy.html

* segment.inventory(reload=False): the `obspy.core.inventory.inventory.Inventory`.
  This object is useful e.g., for removing the instrumental response from `segment.stream()`:
  note that it will be available only if the inventories in xml format were downloaded in
  the downloaded subroutine. As for `stream`, you can also pass a boolean value
  (False by default when missing) to `inventory` to force reloading it from the database.

* segment.siblings(parent=None, condition): returns an iterable of siblings of this segment.
  `parent` can be any of the following:
  - missing or None: returns all segments of the same recorded event, on the
    other channel components / orientations
  - 'stationname': returns all segments of the same station, identified by the tuple of the
    codes (newtwork, station)
  - 'networkname': returns all segments of the same network (network code)
  - 'datacenter', 'event', 'station', 'channel': returns all segments of the same datacenter, event,
    station or channel, all identified by the associated database id.
  `condition` is a dict of expression to filter the returned element. the argument
  `config['segment_select]` can be passed here to return only siblings selected for processing.
  NOTE: Use with care when providing a `parent` argument, as the amount of segments might be huge
  (up to hundreds of thousands of segments). The amount of returned segments is increasing
  (non linearly) according to the following order of the `parent` argument:
  'channel', 'station', 'stationname', 'networkname', 'event' and 'datacenter'

* segment.del_classes(*labels): Deletes the given classes of the segment. The argument is
  a comma-separated list of class labels (string). See configuration file for setting up the
  desired classes.
  E.g.: `segment.del_classes('class1')`, `segment.del_classes('class1', 'class2', 'class3')`

* segment.set_classes(*labels, annotator=None): Sets the given classes on the segment,
  deleting first all segment classes, if any. The argument is
  a comma-separated list of class labels (string). See configuration file for setting up the
  desired classes. `annotator` is a keyword argument (optional): if given (not None) denotes the
  user name that annotates the class.
  E.g.: `segment.set_classes('class1')`, `segment.set_classes('class1', 'class2', annotator='Jim')`

* segment.add_classes(*labels, annotator=None): Same as `segment.set_classes` but does not
  delete segment classes first. If a label is already assigned to the segment, it is not added
  again (regardless of whether the 'annotator' changed or not)

* segment.sds_path(root='.'): Returns the segment's file path in a seiscomp data
  structure (SDS) format:
     <root>/<event_id>/<net>/<sta>/<loc>/<cha>.D/<net>.<sta>.<loc>.<cha>.<year>.<day>
  See https://www.seiscomp3.org/doc/applications/slarchive/SDS.html for details.
  Example: to save the segment's waveform as miniSEED you can type (explicitly
  adding the file extension '.mseed' to the output path):
  ```
      segment.stream().write(segment.sds_path() + '.mseed', format='MSEED')
  ```

* segment.dbsession(): returns the database session for custom IO operations with the database.
  WARNING: this is for advanced users experienced with SQLAlchemy library. If you want to
  use it you probably want to import stream2segment in custom code. See the github documentation
  in case

segment attributes:
-------------------

===================================== ==============================================================
Segment attribute                     Python type and (optional) description
===================================== ==============================================================
segment.id                            int: segment (unique) db id
segment.event_distance_deg            float: distance between the segment's station and
                                      the event, in degrees
segment.event_distance_km             float: distance between the segment's station and
                                      the event, in km, assuming a perfectly spherical earth
                                      with a radius of 6371 km
segment.start_time                    datetime.datetime: the waveform data start time
segment.arrival_time                  datetime.datetime: the station's arrival time of the waveform.
                                      Value between 'start_time' and 'end_time'
segment.end_time                      datetime.datetime: the waveform data end time
segment.request_start                 datetime.datetime: the requested start time of the data
segment.request_end                   datetime.datetime: the requested end time of the data
segment.duration_sec                  float: the waveform data duration, in seconds
segment.missing_data_sec              float: the number of seconds of missing data, with respect
                                      to the requested time window. It might also be negative
                                      (more data received than requested). This parameter is useful
                                      when selecting segments: e.g., if we requested 5
                                      minutes of data and we want to process segments with at
                                      least 4 minutes of downloaded data, then:
                                      missing_data_sec: '< 60'
segment.missing_data_ratio            float: the portion of missing data, with respect
                                      to the request time window. It might also be negative
                                      (more data received than requested). This parameter is useful
                                      when selecting segments: e.g., if you want to process
                                      segments whose real time window is at least 90% of the
                                      requested one, then: missing_data_ratio: '< 0.1'
segment.sample_rate                   float: the waveform data sample rate.
                                      It might differ from the segment channel's sample_rate
segment.has_data                      boolean: tells if the segment has data saved (at least
                                      one byte of data). This parameter is useful when selecting
                                      segments (in most cases, almost necessary), e.g.:
                                      has_data: 'true'
segment.download_code                 int: the code reporting the segment download status. This
                                      parameter is useful to further refine the segment selection
                                      skipping beforehand segments with malformed data (code -2):
                                      has_data: 'true'
                                      download_code: '!=-2'
                                      (All other codes are generally of no interest for the user.
                                      However, for details see Table 2 in
                                      https://doi.org/10.1785/0220180314#tb2)
segment.maxgap_numsamples             float: the maximum gap or overlap found in the waveform data,
                                      in number of points. If 0, the segment has no gaps/overlaps.
                                      Otherwise, if >=1: the segment has gaps, if <=-1: the segment
                                      has overlaps. Values in (-1, 1) are difficult to interpret: a
                                      rule of thumb is to consider half a point a gap / overlap
                                      (maxgap_numsamples > 0.5 or maxgap_numsamples < -0.5).
                                      This parameter is useful when selecting segments: e.g.,
                                      to select segments with no gaps/overlaps, then:
                                      maxgap_numsamples: '(-0.5, 0.5)'
segment.seed_id                       str: the seed identifier in the typical format
                                      [Network].[Station].[Location].[Channel]. For segments
                                      with waveform data, `data_seed_id` (see below) might be
                                      faster to fetch.
segment.data_seed_id                  str: same as 'segment.seed_id', but faster to get because it
                                      reads the value stored in the waveform data. The drawback
                                      is that this value is null for segments with no waveform data
segment.has_class                     boolean: tells if the segment has (at least one) class
                                      assigned
segment.data                          bytes: the waveform (raw) data. Used by `segment.stream()`
------------------------------------- ------------------------------------------------
segment.event                         object (attributes below)
segment.event.id                      int
segment.event.event_id                str: the id returned by the web service or catalog
segment.event.time                    datetime.datetime
segment.event.latitude                float
segment.event.longitude               float
segment.event.depth_km                float
segment.event.author                  str
segment.event.catalog                 str
segment.event.contributor             str
segment.event.contributor_id          str
segment.event.mag_type                str
segment.event.magnitude               float
segment.event.mag_author              str
segment.event.event_location_name     str
------------------------------------- ------------------------------------------------
segment.channel                       object (attributes below)
segment.channel.id                    int
segment.channel.location              str
segment.channel.channel               str
segment.channel.depth                 float
segment.channel.azimuth               float
segment.channel.dip                   float
segment.channel.sensor_description    str
segment.channel.scale                 float
segment.channel.scale_freq            float
segment.channel.scale_units           str
segment.channel.sample_rate           float
segment.channel.band_code             str: the first letter of channel.channel
segment.channel.instrument_code       str: the second letter of channel.channel
segment.channel.orientation_code      str: the third letter of channel.channel
segment.channel.station               object: same as segment.station (see below)
------------------------------------- ------------------------------------------------
segment.station                       object (attributes below)
segment.station.id                    int
segment.station.network               str: the station's network code, e.g. 'AZ'
segment.station.station               str: the station code, e.g. 'NHZR'
segment.station.netsta_code           str: the network + station code, concatenated with
                                      the dot, e.g.: 'AZ.NHZR'
segment.station.latitude              float
segment.station.longitude             float
segment.station.elevation             float
segment.station.site_name             str
segment.station.start_time            datetime.datetime
segment.station.end_time              datetime.datetime
segment.station.has_inventory         boolean: tells if the segment's station inventory has
                                      data saved (at least one byte of data).
                                      This parameter is useful when selecting segments: e.g.,
                                      to select only segments with inventory downloaded:
                                      station.has_inventory: 'true'
segment.station.datacenter            object (same as segment.datacenter, see below)
------------------------------------- ------------------------------------------------
segment.datacenter                    object (attributes below)
segment.datacenter.id                 int
segment.datacenter.station_url        str
segment.datacenter.dataselect_url     str
segment.datacenter.organization_name  str
------------------------------------- ------------------------------------------------
segment.download                      object (attributes below): the download execution
segment.download.id                   int
segment.download.run_time             datetime.datetime
------------------------------------- ------------------------------------------------
segment.classes.id                    int: the id(s) of the classes assigned to the segment
segment.classes.label                 int: the label(s) of the classes assigned to the segment
segment.classes.description           int: the description(s) of the classes assigned to the
                                      segment
===================================== ================================================
'''

from __future__ import division

# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports). UNCOMMENT or REMOVE
# if you are working in Python3 (recommended):
from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)

# From Python >= 3.6, dicts keys are returned (and thus, written to file) in the order they
# are inserted. Prior to that version, to preserve insertion order you needed to use OrderedDict:
from collections import OrderedDict
from datetime import datetime, timedelta  # always useful
from math import factorial  # for savitzky_golay function

# import numpy for efficient computation:
import numpy as np
# import obspy core classes (when working with times, use obspy UTCDateTime when possible):
from obspy import Trace, Stream, UTCDateTime
from obspy.geodetics import degrees2kilometers as d2km
# decorators needed to setup this module @gui.preprocess @gui.plot:
from stream2segment.process import gui
# strem2segment functions for processing obspy Traces. This is just a list of possible functions
# to show how to import them:
from stream2segment.process.math.traces import ampratio, bandpass, cumsumsq,\
    timeswhere, fft, maxabs, utcdatetime, ampspec, powspec, timeof, sn_split
# stream2segment function for processing numpy arrays:
from stream2segment.process.math.ndarrays import triangsmooth, snr
from obspy.signal.spectral_estimation import PPSD, get_nlnm, get_nhnm


def assert1trace(stream):
    '''asserts the stream has only one trace, raising an Exception if it's not the case,
    as this is the pre-condition for all processing functions implemented here.
    Note that, due to the way we download data, a stream with more than one trace his
    most likely due to gaps / overlaps'''
    # stream.get_gaps() is slower as it does more than checking the stream length
    if len(stream) != 1:
        raise ValueError("%d traces (probably gaps/overlaps)" % len(stream))


@gui.preprocess
def bandpass_remresp(segment, config):
    """Removes the offset of the segment's trace, modifies the trace in-place
    :return: a Trace object.
    """
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    trace = stream[0]
    trace.data = trace.data.astype(float) - np.nanmean(trace.data)
    return trace


def signal_noise_spectra(segment, config):
    """
    Computes the signal and noise spectra, as dict of strings mapped to tuples (x0, dx, y).
    Does not modify the segment's stream or traces in-place

    :return: a dict with two keys, 'Signal' and 'Noise', mapped respectively to the tuples
    (f0, df, frequencies)

    :raise: an Exception if `segment.stream()` is empty or has more than one trace (possible
    gaps/overlaps)
    """
    o_trace = segment.stream()[0]
    trace = o_trace  # divide_sensitivity(o_trace, segment.inventory())
    # get sn windows: PLEASE NOTE!! sn_windows might calculate the cumulative of segment.stream(),
    # thus the latter should have been preprocessed (e.g. remove response, bandpass):
    arrival_time = UTCDateTime(segment.arrival_time) + config['sn_windows']['arrival_time_shift']
    signal_trace, noise_trace = sn_split(trace,  # assumes stream has only one trace
                                         arrival_time, config['sn_windows']['signal_window'])

    # compute psd values for both noise and signal:
    psd_n_x, psd_n_y = psd(noise_trace, segment.inventory())
    psd_s_x, psd_s_y = psd(signal_trace, segment.inventory())
    nlnm_x, nlnm_y = get_nlnm()
    nlnm_x, nlnm_y = nlnm_x[::-1], nlnm_y[::-1]
    nhnm_x, nhnm_y = get_nhnm()
    nhnm_x, nhnm_y = nhnm_x[::-1], nhnm_y[::-1]

    # sample at equally spaced periods. First get bounds:
    period_min = 2.0 / trace.stats.sampling_rate
    period_max = min(psd_n_x[-1] - psd_n_x[0], psd_s_x[-1] - psd_s_x[0])

    n_pts = config['num_psd_periods']  # 1024
    periods = np.linspace(period_min, period_max, n_pts, endpoint=True)
    psd_n_y = np.interp(np.log10(periods), np.log10(psd_n_x), psd_n_y)
    psd_s_y = np.interp(np.log10(periods), np.log10(psd_s_x), psd_s_y)
    nlnm_y = np.interp(np.log10(periods), np.log10(nlnm_x), nlnm_y)
    nhnm_y = np.interp(np.log10(periods), np.log10(nhnm_x), nhnm_y)

    x0, dx = periods[0], periods[1] - periods[0]

    return {
        'Signal': (x0, dx, psd_s_y),
        'Noise': (x0, dx, psd_n_y),
        'nlnm': (x0, dx, nlnm_y),
        'nhnm': (x0, dx, nhnm_y)
    }


def divide_sensitivity(trace, inventory):
    '''divides the trace by its sensitivty, returns a copy of the trace.
    Implemented for testing purposes, it is CURRENTLY NOT USED
    '''
    response = inventory.get_response(trace.id, trace.stats.starttime)
    val = response.instrument_sensitivity.value
    trc = trace.copy()
    trc.data = trc.data.astype(float) / val
    return trc


def psd(trace, inventory):
    '''Returns the tuple (psd_x, psd_y) values where the first argument is
    a numopy array of periods and the second argument is a numpy array of
    power spectrum values in Decibel
    '''
    tr = trace
    dt = (tr.stats.endtime.datetime - tr.stats.starttime.datetime).total_seconds()
    ppsd = PPSD(tr.stats, metadata=inventory, ppsd_length=int(dt))
    ppsd.add(tr)
    return ppsd.period_bin_centers, ppsd.psd_values[0]


######################################
# GUI functions for displaying plots #
######################################


@gui.plot
def cumulative(segment, config):
    '''Computes the cumulative of the squares of the segment's trace in the form of a Plot object.
    Modifies the segment's stream or traces in-place. Normalizes the returned trace values
    in [0,1]

    :return: an obspy.Trace

    :raise: an Exception if `segment.stream()` is empty or has more than one trace (possible
    gaps/overlaps)
    '''
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    return cumsumsq(stream[0], normalize=True, copy=False)


# @gui.plot('r', xaxis={'type': 'log'}, yaxis={'type': 'log'})
@gui.plot('r', xaxis={'type': 'log'})  # , yaxis={'type': 'log'})
def sn_spectra(segment, config):
    """
    Computes the signal and noise spectra, as dict of strings mapped to tuples (x0, dx, y).
    Does NOT modify the segment's stream or traces in-place

    :return: a dict with two keys, 'Signal' and 'Noise', mapped respectively to the tuples
    (f0, df, frequencies)

    :raise: an Exception if `segment.stream()` is empty or has more than one trace (possible
    gaps/overlaps)
    """
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    return signal_noise_spectra(segment, config)
