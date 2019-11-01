'''
Created on 1 Nov 2019

@author: riccardo
'''

def open_dataset(filename=None, normalize_=True, verbose=True):
    if filename is None:
        filename = DATASET_FILENAME
    if verbose:
        print('Opening %s' % abspath(filename))

    # capture warnings which are redirected to stderr:
    with capture_stderr(verbose):
        dfr = pd.read_hdf(filename)

        if 'Segment.db.id' in dfr.columns:
            if ID_COL in dfr.columns:
                raise ValueError('The data frame already contains a column '
                                 'named "%s"' % ID_COL)
            # if it's a prediction dataframe, it's for backward compatibility
            dfr.rename(columns={"Segment.db.id": ID_COL}, inplace=True)

        if is_prediction_dataframe(dfr):
            if verbose:
                print('The dataset contains predictions '
                      'performed on a trained classifier. '
                      'Returning the dataset with no further operation')
            return dfr

        if is_station_df(dfr):
            if verbose:
                print('The dataset is per-station basis. '
                      'Returning the dataset with no further operation')
            return dfr

        # setting up columns:
        dfr['pga'] = np.log10(dfr['pga_observed'].abs())
        dfr['pgv'] = np.log10(dfr['pgv_observed'].abs())
        dfr['delta_pga'] = np.log10(dfr['pga_observed'].abs()) - \
            np.log10(dfr['pga_predicted'].abs())
        dfr['delta_pgv'] = np.log10(dfr['pgv_observed'].abs()) - \
            np.log10(dfr['pgv_predicted'].abs())
        del dfr['pga_observed']
        del dfr['pga_predicted']
        del dfr['pgv_observed']
        del dfr['pgv_predicted']
        for col in dfr.columns:
            if col.startswith('amp@'):
                # go to db. We should multuply log * 20 (amp spec) or * 10 (pow spec)
                # but it's unnecessary as we will normalize few lines below
                dfr[col] = np.log10(dfr[col])
        # save space:
        dfr['modified'] = dfr['modified'].astype('category')
        # numpy int64 for just zeros and ones is waste of space: use bools
        # (int8). But first, let's be paranoid first (check later, see below)
        _zum = dfr['outlier'].sum()
        # convert:
        dfr['outlier'] = dfr['outlier'].astype(bool)
        # check:
        if dfr['outlier'].sum() != _zum:
            raise ValueError('The column "outlier" is supposed to be '
                             'populated with zeros or ones, but conversion '
                             'to boolean failed. Check the column')

        if verbose:
            print('')
            print(dfinfo(dfr))

        if normalize_:
            print('')
            dfr = normalize(dfr)

    # for safety:
    dfr.reset_index(drop=True, inplace=True)
    return dfr




