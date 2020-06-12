'''
Created on 20 Apr 2020

@author: riccardo
'''
from os.path import dirname, join, isfile, basename, splitext, abspath
from datetime import timedelta
from sod.jupyter import utils
import pandas as pd
from stream2segment.io.db.models import Station, Segment, Event, Channel
import click
from datetime import datetime

from sqlalchemy.orm import configure_mappers
from sod.core.evaluation import hdf_nrows
configure_mappers()

INPATH = join(dirname(dirname(__file__)), 'datasets', 'sod_dist_2_20_at_rs5.hdf')
OUTPATH = join(dirname(dirname(__file__)), 'datasets', 'sod_dist_2_20_at_rs5_relabelled.hdf')

DATASET_1 = join(dirname(dirname(__file__)), 'datasets', 's2s_2019_03_at_rs5.hdf')
DATASET_2 = join(dirname(dirname(__file__)), 'datasets', 'me_at_rz_minus_vm258.hdf')
 
_cols = ['station_id', 'location_code', 'channel_code', 'event_time',
         'hand_labelled']
 
_grpcols = ['station_id', 'location_code', 'channel_code']

if isfile(OUTPATH):
    raise ValueError('%s\nalready exist, you need to manually delete it to proceed' %
                     OUTPATH)


import logging
logger = logging.getLogger(__name__)
hdlr = logging.FileHandler(abspath(OUTPATH) + '.log', mode='w')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.WARNING)




def warnmsg(sta_id, seed_id, df, msg):
    return ('%s (station_id=%d): %s. '
            '%d of %d instances will have `hand_labelled`=False') % \
            (seed_id, sta_id, msg, (~df.hand_labelled).sum(), len(df))


def run():
    N = 100000

    DF1, DF2 = {}, {}
    for fpath, dsetid, outdict in zip([DATASET_1, DATASET_2],
                                      [1, 2],
                                      [DF1, DF2]):
        print('Reading dataset %d' % dsetid)
        with click.progressbar(length=hdf_nrows(fpath)) as pbar:
            for _df in pd.read_hdf(fpath, columns=_cols, chunksize=N):
                pbar.update(len(_df))
                _df = _df[_df.hand_labelled]
                if _df.empty:
                    continue
                for (s, l, c), _df_ in _df.groupby(_grpcols, sort=False):
                    key = (int(s), str(l), str(c))
                    if key not in outdict:
                        outdict[key] = [_df_.event_time.min(), _df_.event_time.max()]
                    else:
                        tmin, tmax = outdict[key]
                        outdict[key] = [min(tmin, _df_.event_time.min()),
                                        max(tmax, _df_.event_time.max())]

    DF1 = pd.DataFrame({
        'station_id': [_[0] for _ in DF1.keys()],
        'location_code': [_[1] for _ in DF1.keys()],
        'channel_code': [_[2] for _ in DF1.keys()],
        'mintime': [_[0] for _ in DF1.values()],
        'maxtime': [_[1] for _ in DF1.values()]
    })

    DF2 = pd.DataFrame({
        'station_id': [_[0] for _ in DF2.keys()],
        'location_code': [_[1] for _ in DF2.keys()],
        'channel_code': [_[2] for _ in DF2.keys()],
        'mintime': [_[0] for _ in DF2.values()],
        'maxtime': [_[1] for _ in DF2.values()]
    })

    print('Reading current dataset to be relabelled')
    indf = pd.read_hdf(INPATH)
    outdf = []
    sids = pd.unique(indf.station_id).tolist()
    sids2code = {}
    with utils.get_session(4) as sess:
        for (net, sta, sid) in sess.query(Station.network, Station.station, Station.id).\
                filter(Station.id.in_(sids)):
            sids2code[sid] = (net, sta)

    # indf['_tmp'] = indf.location_code.str.cat(indf.channel_code.str[:-1], sep='.')
    gby = indf.groupby(_grpcols, sort=False)
    with click.progressbar(length=len(gby.size())) as pbar:
        for (sid, loc, cha), df in gby:
            sid = int(sid)
            net, sta = sids2code[sid]

            flt = (Station.network == net) & \
                (Station.station == sta) & \
                (Channel.location == loc) & \
                (Channel.channel == cha)

            with utils.get_session(1) as sess:
                _1 = sess.query(Station.id).\
                    join(Station.channels).\
                    filter(flt).all()

            with utils.get_session(2) as sess:
                _2 = sess.query(Station.id).\
                    join(Station.channels).\
                    filter(flt).all()

            seed_id = '.'.join([net, sta, loc, cha])
            # if seed_id == 'IV.ACER..HHN':
            #    asd = 9
            warn = ''
            num_unlabelled = len(df)
            if len(_1) and len(_2):
                warn = 'in both databases'
            elif not len(_1) and not len(_2):
                warn = 'in no databse'
            elif len(_1):
                staids = set(__[0] for __ in _1)
                srcdf = DF1[DF1.station_id.isin(staids) &
                            (DF1.location_code == loc) &
                            (DF1.channel_code == cha)]
            else:
                staids = set(__[0] for __ in _2)
                srcdf = DF2[DF2.station_id.isin(staids) &
                            (DF2.location_code == loc) &
                            (DF2.channel_code == cha)]

            df = df.copy()
            if warn:
                df.hand_labelled = False
            else:
                if srcdf.empty:
                    warn = 'not in source dataset, unlabelling outliers only'
                    df.loc[df.outlier, 'hand_labelled'] = False
                else:
                    mintime, maxtime = srcdf.mintime.min(), srcdf.maxtime.max()
                    # remove label to:
                    # inliers PRIOR to our hand labelling (we can not know if it was
                    # broken, afterwards we assume it is still ok)
                    flt_i = (~df.outlier) & (df.event_time < mintime)
                    # or, for outliers, be stricter as stations might have been
                    # fixed: remove all outliers OUTSIDE the time span where
                    # we sinspected them:
                    flt_o = df.outlier & \
                        ((df.event_time < mintime) | (df.event_time > maxtime)) 
                    df.loc[flt_i | flt_o, 'hand_labelled'] = False
                    num_unlabelled = (~df.hand_labelled).sum()
                    if num_unlabelled:
                        warn = ('some labelled instances outside time bounds '
                                'of source dataset')

            if warn:
                logger.warning(warnmsg(sid, seed_id, df, warn))

            outdf.append(df)
            pbar.update(1)

    outdf = pd.concat(outdf, ignore_index=True, copy=True, axis=0)
    hl = outdf.hand_labelled.sum()
    print('Final result: %d of %d instances hand labbeled' % (hl, len(outdf)))
    key = splitext(basename(OUTPATH))[0].replace('.', '_')
    outdf.to_hdf(OUTPATH, format='table', mode='w', key=key)
    print('Written to "%s"' % OUTPATH)


# def run():
#     indf = pd.read_hdf(INPATH)
#     outdf = []
#     gby = indf.groupby(['station_id', 'location_code', 'channel_code'], sort=False)
#     with click.progressbar(length=len(gby.size())) as pbar:
#         for (sid, lc, cc), df in gby:
#             sid = int(sid)
#             with utils.get_session(4) as sess:
#                 _ = sess.query(Station.network, Station.station).\
#                     filter(Station.id == sid).first()
#                 seed_id = '.'.join([_[0], _[1], lc, cc])
#     
#             with utils.get_session(1) as sess:
#                 _1 = sess.query(Segment.id).\
#                     filter(Segment.data_seed_id == seed_id).limit(1).all()
#     
#             with utils.get_session(2) as sess:
#                 _2 = sess.query(Segment.id).\
#                     filter(Segment.data_seed_id == seed_id).limit(1).all()
#     
#             if len(_1) and len(_2):
#                 print('Warning: %s (station_id=%d) in both database: '
#                       'set hand_labelled to False' % (seed_id, sid))
#                 df = df.copy()
#                 df.hand_labelled = False
#                 outdf.append(df)
#                 continue
#     
#             if not len(_1) and not len(_2):
#                 print('Warning: %s (station_id=%d) not found in any database: '
#                       'set hand_labelled to False' % (seed_id, sid))
#                 df = df.copy()
#                 df.hand_labelled = False
#                 outdf.append(df)
#                 continue
#     
#             dbid = 1 if len(_1) else 2
#             with utils.get_session(dbid) as sess:
#                 times = []
#                 for (_, evtime) in sess.query(Segment.id, Event.time).\
#                         join(Segment.event).filter(Segment.data_seed_id == seed_id):
#                     times.append(evtime)
#                 times = sorted(times)
#                 mintime, maxtime = times[0], times[-1]
#                 assert mintime <= maxtime
#                 assert isinstance(mintime, datetime)
#     
#             df = df.copy()
#             df.loc[(df.event_time < mintime) | (df.event_time > maxtime), 'hand_labelled'] = False
#             no_hl = (~df.hand_labelled).sum()
#             if no_hl:
#                 print('Warning: %s (station_id=%d) '
#                       'has %d segments outside bounds: set %d hand_labelled to False' %
#                       (seed_id, sid, no_hl, no_hl))
#             
#             outdf.append(df)
#             pbar.update(1)
#     
#     outdf = pd.concat(outdf, ignore_index=True, copy=True, axis=0)
#     key = splitext(basename(OUTPATH))[0].replace('.', '_')
#     outdf.to_hdf(OUTPATH, format='table', mode='w', key=key)
#     print('Written to "%s"' % OUTPATH)

if __name__ == '__main__':
    run()