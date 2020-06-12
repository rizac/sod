'''
Created on 3 Apr 2020

@author: riccardo
'''
import os
from os.path import join, isdir, isfile, abspath, expanduser, basename, dirname
from sod.core import paths
from mock import patch
from sod.core.evaluation import _get_summary_evaluationmetrics_mp, save_df
import numpy as np
from pandas import DataFrame, read_hdf
import click
from multiprocessing import Pool, cpu_count
import sys

if __name__ == '__main__':
    nextcloud = join(expanduser('~'), 'Nextcloud', 'rizac', 'outliers_paper' )
    assert isdir(nextcloud)
    filenames = [join(nextcloud, _) for _ in ['lista_1_out_dino',
                                              'lista_2_out_dino']]
    assert all(isfile(_) for _ in filenames)
    ids1, ids2 = [], [] 
    for _, out in zip(filenames, [ids1, ids2]):
        with open(_) as opn:
            for val in opn.read().strip().split(' '):
                out.append(int(val))
    print('ids1: %d segments' % len(ids1))
    print('ids2: %d segments' % len(ids2))

    assert 21202165 in ids1

    evaldir = paths.EVALUATIONS_RESULTS_DIR
    eval_df_path = join(evaldir, 'summary_evaluation_corrected.hdf')
    if isfile(eval_df_path):
        print('File exist, remove manually before proceeding:')
        print(eval_df_path)
        sys.exit(1)
    
    print('Saving to:\n"%s"' % eval_df_path)

    datasetname = 'allset_unlabelled_annotation2.hdf'
    pred_df_paths = []
    for _ in os.listdir(evaldir):
        _ = join(evaldir, _)
        if isdir(_):
            pred_df_paths.extend(join(_, _2) for _2 in os.listdir(_)
                                 if datasetname in _2)
    assert all(isfile(_) for _ in pred_df_paths)
    print("%d prediction dataframes found" % len(pred_df_paths))

    ids1 = np.array(ids1, dtype=int)
    ids2 = np.array(ids2, dtype=int)

    def read_remove(filepath, columns):
        dfr = read_hdf(filepath)
        flt = ((dfr.dataset_id == 1) & (dfr['Segment.db.id'].isin(ids1))) | \
            ((dfr.dataset_id == 2) & (dfr['Segment.db.id'].isin(ids2)))
        return dfr[~flt][columns]

    with patch('sod.core.evaluation.pd.read_hdf', side_effect=read_remove):
        pool = Pool(processes=int(cpu_count()))
        newrows = []
        errors = []
        with click.progressbar(length=len(pred_df_paths),
                               fill_char='o', empty_char='.') as pbar:
            iter_ = (dir)
            for clfdir, testname, dic in \
                    pool.imap_unordered(_get_summary_evaluationmetrics_mp,
                                        ((dirname(_), basename(_)) for _ in pred_df_paths)):
#                     map(_get_summary_evaluationmetrics_mp,
#                                         ((dirname(_), basename(_)) for _ in pred_df_paths)):
                pbar.update(1)
                if isinstance(dic, Exception):
                    errors.append(dic)
                else:
                    dic['_key'] = join(basename(clfdir), testname)
                    newrows.append(dic)

        if newrows:
            dfr = DataFrame(data=newrows)
            save_df(dfr, eval_df_path)
        

