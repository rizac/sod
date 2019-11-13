'''
Created on 16 Aug 2019

@author: riccardo
'''
import os
import yaml
import dateutil.parser

from stream2segment.process.db import get_session
from stream2segment.io.db.models import Station
from stream2segment.io.utils import loads_inv

root = os.path.dirname(__file__)
with open(os.path.join(root, '..', '..', 'jupyter', 'jnconfig.yaml')) as _:
    config = yaml.safe_load(_)
dbpath_old = config['dbpath_old']
dbpath_new = config['dbpath_new']
inv_outdir = os.path.join(root, 'inventories', dbpath_old.split('/')[-1])
inventories = [
    'CH.GRIMS.2011-11-09T00:00:00.xml',
    'CH.GRIMS.2015-10-30T10:50:00.xml',
    'SK.MODS.2004-03-17T00:00:00.xml',
    "SK.ZST.2004-03-17T00:00:00.xml",
    "FR.PYLO.2010-01-17T10:00:00.xml"
]

if __name__ == '__main__':
    sess = get_session(dbpath_old)
    try:
        for invfile in inventories:
            net, sta, stime, ext = invfile.split('.')
            stime = dateutil.parser.parse(stime)
            inv_xml = sess.query(Station.inventory_xml).filter((Station.network == net) &
                                                               (Station.station == sta) &
                                                               (Station.start_time == stime)
                                                               ).all()
            if len(inv_xml) != 1:
                raise Exception("%s: %d inventories found" % (invfile, len(inv_xml)))
            inv_obj = loads_inv(inv_xml[0][0])
            outinv_path = os.path.join(inv_outdir, invfile)
            if not os.path.isdir(os.path.dirname(outinv_path)):
                os.makedirs(os.path.dirname(outinv_path))
            inv_obj.write(outinv_path, "STATIONXML")
            print('Saved %s' % outinv_path)
        
    finally:
        sess.close()
