'''
Created on 12 Jun 2020

@author: riccardo
'''
from os.path import expanduser, basename, isfile, join
import sys
import pandas as pd
from joblib import dump, load

fle = '/Users/riccardo/work/gfz/projects/sources/python/sod/sod/datasets/me_at_rz_minus_vm258.hdf'
mdl = ('/Users/riccardo/work/gfz/projects/sources/python/sod/sod/evaluations/results/'
     'clf=IsolationForest&tr_set=uniform_train.hdf&feats=psd@5sec&behaviour=new&contamination=auto&max_samples=1024&n_estimators=100&random_state=11.sklmodel')

if __name__ == '__main__':
    outfile = join(expanduser('~'), 'Desktop', 'scores.' + basename(fle))
    if isfile(outfile):
        print(f'File exists: "{outfile}"')
        sys.exit(1)
    print()
    print(f'Reading "{fle}"')
    dset = pd.read_hdf(fle, columns=['Segment.db.id', 'psd@5sec', 'window_type'])
    _len = len(dset)
    dset = dset[dset.window_type & (~pd.isna(dset['psd@5sec']))]
    del dset['window_type']
    print(f'{len(dset)} instances found (after removing {_len - len(dset)} noise segments or with NaN feature(s))')
    assert sorted(dset.columns) == ['Segment.db.id', 'psd@5sec']
    features = dset['psd@5sec'].values.reshape((len(dset), 1))
    print()
    print(f'Loading "{mdl}"')
    model = load(mdl)
    print('Model:')
    print(str(model))
    print()
    print(f'Calculating scores for {basename(fle)}')
    zcores = -model.score_samples(features)
    dset['predicted_anomaly_score'] = zcores
    print()
    print(f'Saving to "{outfile}"')
    dset.to_hdf(outfile, mode='w', format='table', key='me_scores')
    print()
    print('Just a check')
    feat, scr = dset["psd@5sec"], dset.predicted_anomaly_score
    print(f'Mean score for psd@5sec in [-125, -100]'
          f'{dset[(feat>-125) & (feat<-100)].predicted_anomaly_score.mean()}')
    print(f'Mean score for psd@5sec in [-150, -125]'
          f'{dset[(feat>-150) & (feat<-125)].predicted_anomaly_score.mean()}')
    print(f'Mean score for psd@5sec in [-100, -75]'
          f'{dset[(feat>-100) & (feat<-75)].predicted_anomaly_score.mean()}')
    print(f'Mean score for psd@5sec < -150'
          f'{dset[(feat<=-150)].predicted_anomaly_score.mean()}')
    print(f'Mean score for psd@5sec > -75'
          f'{dset[(feat>-75)].predicted_anomaly_score.mean()}')
