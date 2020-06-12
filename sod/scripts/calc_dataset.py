'''
Created on 3 May 2020

@author: riccardo
'''
from os.path import join, isfile, abspath, dirname, basename, splitext
from click import progressbar
from os import listdir, sep
from joblib import dump, load
import pandas as pd

from sod.core.paths import EVALUATIONS_RESULTS_DIR, DATASETS_DIR
from sod.core.evaluation import predict
from sod.core.metrics import average_precision_score

TRAINSETNAME = 'uniform_test.hdf'


if __name__ == '__main__':
    files = []
    for _ in listdir(EVALUATIONS_RESULTS_DIR):
        _ = abspath(join(EVALUATIONS_RESULTS_DIR, _, TRAINSETNAME))
        if isfile(_):
            files.append(_)

#     clfs = []
#     print('Loading %d classifiers' % len(files))
#     with progressbar(length=len(files)) as pbar:
#         for f in files:
#             clfs.append(load(dirname(f) + '.sklmodel'))
#             pbar.update(1)
# 
#     print('Loading "%s"' % basename(inpath))
#     indf = pd.read_hdf(inpath, columns=['predicted_anomaly_score', 'outlier',
#                                         'dataset_id'])
#     indf = indf[indf.dataset_id==1].copy()

    evalpath = join(EVALUATIONS_RESULTS_DIR, 'evaluationmetrics.hdf')
    print('Loading "%s"' % evalpath)
    evaldf = pd.read_hdf(evalpath)
    
    print('Recomputing predicitons')
    aps = 'average_precision_score'
    append = [evaldf]
    with progressbar(length=len(files)) as pbar:
        for file in files:
            pbar.update(1)
            key = join(basename(dirname(file)), TRAINSETNAME)
            # check if we have it:
            _ = evaldf.loc[evaldf._key.str.endswith(key)]
            assert len(_) == 1
            _ = _.copy().reset_index(drop=True)
            pred_df = pd.read_hdf(file, columns=['predicted_anomaly_score',
                                                 'outlier',
                                                 'dataset_id'])
            pred_df = pred_df[pred_df.dataset_id == 1].copy()
            aps = average_precision_score(pred_df)
            assert aps != _.loc[0, 'average_precision_score']
            _.loc[0, 'average_precision_score'] = aps
            newname = splitext(key)[0] + "_dataset1.hdf"
            assert _.loc[0, '_key'] == newname.replace("_dataset1", "")
            _.loc[0, '_key'] = newname
            append.append(_)
    destdf = pd.concat(append, axis=0, sort=False, ignore_index=True,
                       copy=True)
    print('Writing to "%s"' % evalpath)
    destdf.to_hdf(evalpath, mode='w', format='table', key='evaluationmetrics')
