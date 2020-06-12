'''
Created on 4 Apr 2020

@author: riccardo
'''
from sod.core.paths import EVALUATIONS_RESULTS_DIR
from os.path import join, isdir, isfile, getmtime, dirname, basename, abspath, splitext
import click
from os import listdir, stat, makedirs, rename, mkdir
from time import ctime
import sys

FILENAME = 'uniform_test_dataset1.hdf'
DESTDIR = abspath(join(dirname(__file__), 'pred_dfs_deleted'))
assert isdir(dirname(DESTDIR))
if not isdir(DESTDIR):
    mkdir(DESTDIR)

if __name__ == '__main__':
    edir = EVALUATIONS_RESULTS_DIR
    filenames = []
    allfiles = set()
    for dir_ in (join(edir, _) for _ in listdir(edir) if isdir(join(edir, _))):
        for fle in listdir(dir_):
            if fle == FILENAME:
                infile = join(dir_, fle)
                assert isfile(infile)
                filenames.append(infile)
            if splitext(fle)[1] != '.hdf':
                asd = 9
            allfiles.add(fle)
    print('All predictions (HDF) found are: %s' % str(allfiles))
    maxtime = -10
    for f in filenames:
        tme = getmtime(f)
        if tme > maxtime:
            maxtime = tme
    print()
    print('%d files named "%s" found' % (len(filenames), FILENAME))
    if len(filenames) == 0:
        sys.exit(0)
    print('The more recent was modified on: %s' % str(ctime(maxtime)))
    print()
    resp = input('Do you want to MOVE those predictions to\n"%s"?\n(y/n)' %
                 DESTDIR)
    if resp != 'y':
        print('Aborted')
        sys.exit(1)
    for f in filenames:
        destfile = join(DESTDIR, basename(dirname(f)), basename(f))
        if not isdir(dirname(destfile)):
            mkdir(dirname(destfile))
        rename(f, destfile)
