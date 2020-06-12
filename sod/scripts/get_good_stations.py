'''
Created on 7 Apr 2020

@author: riccardo
'''
from datetime import datetime, timedelta
from stream2segment.process.db import get_session, Station, Channel
import pandas as pd
from os.path import join, dirname, isdir, join, expanduser, basename, relpath
import yaml
from itertools import chain
import re


def yamlload(filepath):
    with open(filepath) as stream:
        return yaml.safe_load(stream)


def yamldump(filepath, dic, **kw):
    kw.setdefault('default_flow_style', False)
    kw.setdefault('sort_keys', False)
    print(f'Dumping YAML to "{relpath(filepath)}"')
    with open(filepath, 'w') as stream:
        yaml.safe_dump(dic, stream, **kw)


_dbpaths = join(dirname(dirname(__file__)), 'jupyter', 'jnconfig.yaml')
db_src_1 = yamlload(_dbpaths)['dbpath_eu_new']
# 'postgresql://rizac:***@rs5.gfz-potsdam.de/s2s_2019_03'
db_src_2 = yamlload(_dbpaths)['dbpath_me']
# 'postgresql://me:***@rz-vm258.gfz-potsdam.de/me'

# root configurations of the feature extraxtion modules, where
# to take the inliers/outliers:
_rootconfigs = join(dirname(dirname(__file__)), 'stream2segment', 'configs')
yaml_src_1 = join(_rootconfigs, 's2s_2019_03_at_rs5.yaml')
yaml_src_2 = join(_rootconfigs, 'me_at_rz_minus_vm258.yaml')

yaml_download_sod_dist_2_20_template = join(_rootconfigs,
                                            'downloads',
                                            'sod_dist_2_20_at_rs5.TEMPLATE.yaml')

yaml_download_sod_mag_4_5_template = join(_rootconfigs,
                                          'downloads',
                                          'sod_mag_4_5_at_rs5.TEMPLATE.yaml')

yaml_process_sod_dist_2_20_template = join(_rootconfigs,
                                           'sod_dist_2_20_at_rs5.yaml')

yaml_process_sod_mag_4_5_template = join(_rootconfigs,
                                         'sod_mag_4_5_at_rs5.yaml')


def get_outliers_from_annotations():
    '''gets the outliers annotated by experts, returns two lists:
    the first referring to dataset_id 1, the second to dataset_id 2.
    Every element of the lists is str in the form N.S.L.C
    '''
    outliers_1, outliers_2 = [], []
    dinoannotroot = join(expanduser('~'), 'Nextcloud', 'rizac',
                         'outliers_paper')
    for filename in ['0.55_0.65_dino.csv', '0.65_0.75_dino.csv',
                     '0.75_0.85_dino.csv']:
        csv = pd.read_csv(join(dinoannotroot, filename))
        csv = csv[pd.isna(csv.start_time) & pd.isna(csv.end_time) &
                  pd.to_numeric(csv.score_d, errors='coerce') == 1]
        if csv.empty:
            continue
        outliers_1.append(csv[csv.dataset_id == 1][['station_id', 'location.channel']])
        outliers_2.append(csv[csv.dataset_id == 2][['station_id', 'location.channel']])

    outliers_1 = pd.concat(outliers_1, axis=0, sort=False)
    outliers_2 = pd.concat(outliers_2, axis=0, sort=False)

    # now we have two dataframes with columns 'station_id' (int)
    # and 'location.channel' (str). we need to convert the first to
    # "net.sta" string
    for (dburl, dfr) in [
            (
                db_src_1,
                outliers_1
            ),
            (
                db_src_2,
                outliers_2
            )
    ]:
        sess = get_session(dburl)
        try:
            dfr['network.station'] = ''
            for (sid, net, sta) in sess.query(Station.id, Station.network,
                                              Station.station).\
                    filter(Station.id.in_(dfr.station_id.tolist())):
                dfr.loc[dfr.station_id == sid, 'network.station'] = net + '.' + sta
        finally:
            sess.close()

    outliers_1 = outliers_1['network.station'].\
        str.cat(outliers_1['location.channel'], sep='.')
    outliers_2 = outliers_2['network.station'].\
        str.cat(outliers_2['location.channel'], sep='.')

    return sorted(set(outliers_1)), sorted(set(outliers_2))


def write_configs(inliers_1, outliers_1, inliers_2, outliers_2):

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    print()
    print(f'Writing download configs')
    yaml_dic = yamlload(yaml_download_sod_dist_2_20_template)

    # save download seisms not included (by distances) for s2s_2019_03:
    dic = dict(yaml_dic)
    dic['starttime'] = today.replace(year=today.year-15).isoformat()
    dic['endtime'] = today.isoformat()
    dic['network'] = sorted(set(_.split('.')[0] for _ in
                                chain(inliers_1, outliers_1)))
    dic['station'] = sorted(set(_.split('.')[1] for _ in
                                chain(inliers_1, outliers_1)))
    dic['channel'] = sorted(set(_.split('.')[3] for _ in
                                chain(inliers_1, outliers_1)))
    dic['dataws'] = 'eida'
    fname = yaml_download_sod_dist_2_20_template.replace('.TEMPLATE.yaml',
                                                         '.s2s_2019_03.yaml')
    yamldump(fname, dic)

    # save download seisms not included (by distances) for me:
    dic = dict(yaml_dic)
    dic['starttime'] = today.replace(year=today.year-15).isoformat()
    dic['endtime'] = today.isoformat()
    dic['network'] = sorted(set(_.split('.')[0] for _ in
                                chain(inliers_2, outliers_2)))
    dic['station'] = sorted(set(_.split('.')[1] for _ in
                                chain(inliers_2, outliers_2)))
    dic['channel'] = sorted(set(_.split('.')[3] for _ in
                                chain(inliers_2, outliers_2)))

    fname = yaml_download_sod_dist_2_20_template.replace('.TEMPLATE.yaml',
                                                         '.me.eida.yaml')
    dic['dataws'] = 'eida'
    print(f'writing to {basename(fname)}')
    yamldump(fname, dic)

    fname = yaml_download_sod_dist_2_20_template.replace('.TEMPLATE.yaml',
                                                         '.me.iris.yaml')
    dic['dataws'] = 'iris'
    print(f'writing to {basename(fname)}')
    yamldump(fname, dic)

    # save download seism not included (by mag) for me:
    yaml_dic = yamlload(yaml_download_sod_mag_4_5_template)
    dic = dict(yaml_dic)
    dic['starttime'] = today.replace(year=today.year-15).isoformat()
    dic['endtime'] = today.isoformat()
    dic['network'] = sorted(set(_.split('.')[0] for _ in
                                chain(inliers_2, outliers_2)))
    dic['station'] = sorted(set(_.split('.')[1] for _ in
                                chain(inliers_2, outliers_2)))
    dic['channel'] = sorted(set(_.split('.')[3] for _ in
                                chain(inliers_2, outliers_2)))
    dic['dataws'] = 'iris'
    fname = yaml_download_sod_mag_4_5_template.replace('.TEMPLATE.yaml',
                                                       '.me.iris.yaml')
    yamldump(fname, dic)

    print()
    print('Writing s2s process config files')
    # need to write to text cause we want to preserve comments:
    with open(yaml_process_sod_dist_2_20_template) as opn:
        content = opn.read()
        good = "\n".join('  "%s": null' % _
                           for _ in sorted(set(inliers_1 + inliers_2)))
        content = re.sub('\ngood:.*?\n\n',
                         "\ngood:\n" + good + "\n\n",
                         content,
                         re.DOTALL)
        bad = "\n".join('  "%s": null' % _
                        for _ in sorted(set(outliers_1 + outliers_2)))
        content = re.sub('\nbad:.*?\n\n',
                         "\nbad:\n" + bad + "\n\n",
                         content,
                         re.DOTALL)
    with open(yaml_process_sod_dist_2_20_template, 'w') as opn:
        opn.write(content)

    with open(yaml_process_sod_mag_4_5_template) as opn:
        content = opn.read()
        good = "\n".join('  "%s": null' % _
                           for _ in sorted(set(inliers_2)))
        content = re.sub('\ngood:.*?\n\n',
                         "\ngood:\n" + good + "\n\n",
                         content,
                         re.DOTALL)
        bad = "\n".join('  "%s": null' % _
                        for _ in sorted(set(outliers_2)))
        content = re.sub('\nbad:.*?\n\n',
                         "\nbad:\n" + bad + "\n\n",
                         content,
                         re.DOTALL)
    with open(yaml_process_sod_mag_4_5_template, 'w') as opn:
        opn.write(content)


if __name__ == '__main__':

    # dino annotations:
    outliers_1, outliers_2 = get_outliers_from_annotations()

    sess = get_session(db_src_1)
    try:
        # 'good' keys are already in the format N.S.L.C
        with open(yaml_src_1) as opn:
            inliers_1 = list(yamlload(yaml_src_1)['good'].keys())
    finally:
        sess.close()

    sess = get_session(db_src_2)
    try:
        inliers_2 = []
        # good are station ids:
        good_sta_ids = list(yamlload(yaml_src_2)['good'].keys())
        for (c, n, s) in sess.query(Channel, Station.network, Station.station).\
            join(Channel.station).\
                filter(Station.id.in_(list(good_sta_ids))):
            inliers_2.append(f'{n}.{s}.{c.location}.{c.channel}')
    finally:
        sess.close()

    inliers_1 = sorted(set(inliers_1))
    inliers_2 = sorted(set(inliers_2))
    outliers_1 = sorted(set(outliers_1))
    outliers_2 = sorted(set(outliers_2))

    print()
    print(f'{db_src_1}'
          f'\ninliers ({len(inliers_1)}):\n{inliers_1}'
          f'\noutliers ({len(outliers_1)}):\n{outliers_1}')
    print()
    print(f'{db_src_2}'
          f'\ninliers ({len(inliers_2)}):\n{inliers_2}'
          f'\noutliers ({len(outliers_2)}):\n{outliers_2}')

    write_configs(inliers_1, outliers_1, inliers_2, outliers_2)

#                 inliers_1 = list()
#                 for good in good_sta_ids:
#                     _ = good.split('.')
#                     net.add(_[0])
#                     sta.add(_[1])
#                     cha.add(_[3])
#                 # write to eida:
#                 dic = dict(yaml_dict)
#                 dic['network'] = list(net)
#                 dic['station'] = list(sta)
#                 dic['channel'] = list(cha)
#                 dic['dataws'] = 'eida'
#                 fileout = join(dirname(template_yaml), 'sod_dist_2_20_at_rs5.eida.yaml')
#                 with open(fileout, 'w') as stream:
#                     yaml.safe_dump(dic, stream, default_flow_style=False, sort_keys=False)
