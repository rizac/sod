import numpy as np
from obspy.signal.spectral_estimation import PPSD, dtiny
from obspy.core.stream import read
from os.path import dirname, join
from obspy.core.inventory.inventory import read_inventory
from matplotlib import mlab
import math
import time


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


if __name__ == "__main__":
    trace, inv = 'trace_GE.APE.mseed', 'inventory_GE.APE.xml'
    stream = read(join(dirname(__file__), 'miniseed', trace))
    inv = read_inventory(join(dirname(__file__), 'miniseed', inv))
    t = time.time()
    psd_values([5], stream[0], inv)
    
    
    
def __process(tr, metadata, special_handling=None):
    """
    Processes a segment of data and save the psd information.
    Whether `Trace` is compatible (station, channel, ...) has to
    checked beforehand.

    :type tr: :class:`~obspy.core.trace.Trace`
    :param tr: Compatible Trace with data of one PPSD segment
    :returns: `True` if segment was successfully processed,
        `False` otherwise.
    """
#     # XXX DIRTY HACK!!
#     if len(tr) == self.len + 1:
#         tr.data = tr.data[:-1]
#     # one last check..
#     if len(tr) != self.len:
#         msg = "Got a piece of data with wrong length. Skipping"
#         warnings.warn(msg)
#         print(len(tr), self.len)
#         return False
#     # being paranoid, only necessary if in-place operations would follow
#     tr.data = tr.data.astype(np.float64)

    # if trace has a masked array we fill in zeros
    try:
        tr.data[tr.data.mask] = 0.0
    # if it is no masked array, we get an AttributeError
    # and have nothing to do
    except AttributeError:
        pass

    # restitution:
    # mcnamara apply the correction at the end in freq-domain,
    # does it make a difference?
    # probably should be done earlier on bigger chunk of data?!
    # Yes, you should avoid removing the response until after you
    # have estimated the spectra to avoid elevated lp noise

    spec, _freq = mlab.psd(tr.data, self.nfft, self.sampling_rate,
                           detrend=mlab.detrend_linear, window=fft_taper,
                           noverlap=self.nlap, sides='onesided',
                           scale_by_freq=True)

    # leave out first entry (offset)
    spec = spec[1:]

    # working with the periods not frequencies later so reverse spectrum
    spec = spec[::-1]

    # Here we remove the response using the same conventions
    # since the power is squared we want to square the sensitivity
    # we can also convert to acceleration if we have non-rotational data
    if self.special_handling == "ringlaser":
        # in case of rotational data just remove sensitivity
        spec /= self.metadata['sensitivity'] ** 2
    # special_handling "hydrophone" does instrument correction same as
    # "normal" data
    else:
        # determine instrument response from metadata
        try:
            resp = self._get_response(tr)
        except Exception as e:
            msg = ("Error getting response from provided metadata:\n"
                   "%s: %s\n"
                   "Skipping time segment(s).")
            msg = msg % (e.__class__.__name__, str(e))
            # warnings.warn(msg)
            # return False
            raise ValueError(msg)

        resp = resp[1:]
        resp = resp[::-1]
        # Now get the amplitude response (squared)
        respamp = np.absolute(resp * np.conjugate(resp))
        # Make omega with the same conventions as spec
        w = 2.0 * math.pi * _freq[1:]
        w = w[::-1]
        # Here we do the response removal
        # Do not differentiate when `special_handling="hydrophone"`
        if self.special_handling == "hydrophone":
            spec = spec / respamp
        else:
            spec = (w ** 2) * spec / respamp
    # avoid calculating log of zero
    idx = spec < dtiny
    spec[idx] = dtiny

    # go to dB
    spec = np.log10(spec)
    spec *= 10

    smoothed_psd = []
    # do this for the whole period range and append the values to our lists
    for per_left, per_right in zip(self.period_bin_left_edges,
                                   self.period_bin_right_edges):
        specs = spec[(per_left <= self.psd_periods) &
                     (self.psd_periods <= per_right)]
        smoothed_psd.append(specs.mean())
    smoothed_psd = np.array(smoothed_psd, dtype=np.float32)
    self.__insert_processed_data(tr.stats.starttime, smoothed_psd)
    return True