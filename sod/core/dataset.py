'''
Datasets (hdf files) module with IO operations, classes definition and more

Created on 1 Nov 2019

@author: riccardo
'''
import sys
from os.path import (splitext, dirname, join, basename, isfile, isdir, isabs,
                     abspath)
from io import StringIO
from contextlib import contextmanager
import numpy as np
import pandas as pd

from sod.core import pdconcat, odict
from sod.core.paths import DATASETS_DIR

#####################
# CLASSES DEFINITIONS
#####################

# FOR ANY NEW DATASET:
# 1. IMPLEMENT THE YAML, AND PY, LAUNCH S2S AND SAVE THE DATAFRAME WITH A
#    NEW UNIQUE NAME <NAME>.HDF
# 2. IF IT HAS DIFFERENT UNIQUE COLUMNS THAN THE PREVIOUSLY DEFINED DATASETS,
#    ADDS THE UNIQUE COLUMNS BELOW (THIS WILL BE USEFUL TO GET VIA SCRIPTS
#    WHICH SEGMENT CORRESPOND TO A PREDICTION IN A PREDICTION DATAFRAME)
# 3. IF IT HAS DIFFERENT CLASSES DEFINITIONS, ADD THEM TO _CLASSES AND THEN
#    DEFINE THE RELATIVE if BRANCH IN `classes_of` (THIS WILL BE USEFUL WHEN
#    CORRECTLY DISPLAYING CONFUSION MATRICES)

# column names definition
# ID_COL = 'id'  # every read operation will convert 'Segment.db.id' into this
OUTLIER_COL = 'outlier'  # MUST be defined in all datasets


def is_outlier(dataframe):
    '''pandas series of boolean telling where dataframe rows are outliers'''
    return dataframe['outlier']  # simply return the column


##########################
# dataset IO function(s) #
##########################


def open_dataset(filename, normalize=True, verbose=True):

    filepath = dataset_path(filename)
    keyname = splitext(basename(filepath))[0]

    try:
        datasetinfo = globals()[keyname]
    except KeyError:
        raise ValueError('Invalid dataset, no function "%s" '
                         'implemented' % keyname)

    return datasetinfo.open(filename, normalize, verbose)


def dataset_path(filename, assure_exist=True):
    keyname, ext = splitext(filename)
    if not ext:
        filename += '.hdf'
    filepath = abspath(join(DATASETS_DIR, filename))
    if assure_exist and not isfile(filepath):
        raise ValueError('Invalid dataset, File not found: "%s"'
                         % filepath)
    return filepath

#################
# Functions mapped to specific datasets in 'datasets' and performing
# custom dataframe operations
#################


def dataset_info(dataframe):
    col = dataframe.columns[0]
    if col.endswith('.id'):
        return globals()[col[:col.index('.id')]]
    raise ValueError('DataFrame not bound to a known dataset')


class UIDcolsSetter(type):
    '''Metaclass for the DatasetInfo class'''
    def __new__(cls, name, bases, dct):
        '''basically modifies cls.uid_columns'''
        newcls = super().__new__(cls, name, bases, dct)
        _uidcols = [_ for _ in newcls._uid_columns if _ != OUTLIER_COL]
        newcls.uid_columns = tuple([name + '.id'] +
                                   _uidcols +
                                   [OUTLIER_COL])
        
        
#         uidcols = list(newcls.uid_columns)
#         if OUTLIER_COL not in uidcols:
#             uidcols += [OUTLIER_COL]
#         if hasattr(newcls, '_classcol'):
#             uidcols[0] = name+'.id'
#         else:
#             uidcols.insert(0, name+'.id')
#         newcls._classcol = uidcols[0]
#         newcls.uid_columns = tuple(uidcols)
        return newcls


class DatasetInfo(metaclass=UIDcolsSetter):

    S2S_COL = 'Segment.db.id'

    #########################################
    # MEMBERS TO BE OVERWRITTEN IN SUBCLASSES
    #########################################

    # tuple of unique columns necessary to identify a Data Fraame as belonging
    # to this dataset. Note that any function generating derived Data Frame
    # must also pass these columns to the generated Data frames.
    # (See e.g., `evaluation.predict`, which will save these columns in
    # addition to the produced prediction scores and decision functions).
    # In `Meta` (which is called when creating this class), the FIRST element
    # WILL ALWAYS be an id denoting the dataset (classname+ '.id') and will
    # replace the S2S_COL of the dataframe (see `open_dataset`).
    # The column OUTLIER_COL will also be added in `Meta` and needs to be
    # specified
    _uid_columns = tuple()  # (S2S_COL,)

    # tuple of this dataset (sub)classes:
    classnames = tuple()

    # dict where each dataset's subclass is mapped to a selector function:
    # "func(dataframe)" returning the pandas Series of booleans indicating
    # where the dataframe row match the given class, so that, to filter
    # the dataframe with class rows only you call: dataframe[func(dataframe)]
    class_selector = {}

    # dict where each dataset's class is mapped to its weight. The weight is
    # only used in html evaluation reports to dynamically sort conf.matrices
    class_weight = {}

    ############################################

    @classmethod
    def open(cls, filename, normalize=True, verbose=True):

        filepath = dataset_path(filename)
        keyname = splitext(basename(filepath))[0]

        if verbose:
            print('Opening %s' % abspath(filepath))

        # capture warnings which are redirected to stderr:
        with capture_stderr(verbose):
            dfr = pd.read_hdf(filepath)

            # EXTREMELY IMPORTANT PART: create the id column and set it
            idcol = cls.uid_columns[0]
            dfr.insert(0, idcol, dfr[cls.S2S_COL])
            dfr.drop(cls.S2S_COL, axis=1, inplace=True)

            try:
                dfr = cls._open(dfr)
            except Exception as exc:
                raise ValueError('Check module function "%s", error: %s' %
                                 (cls.__name__, str(exc)))

            kount = 0
            for cname in cls.classnames:
                _ = dfr[cls.class_selector[cname](dfr)]
                kount += len(_)
                sum_ = _[OUTLIER_COL].sum()
                if sum_ > 0 and sum_ != len(_):
                    raise ValueError('subclass "%s" contains both inliers '
                                     'and outliers. Please change the '
                                     'class selectors in the code' % cname)
            if kount != len(dfr):
                raise ValueError('Rows count by subclasses does '
                                 'not sum up to total number of rows '
                                 'Please change the '
                                 'class selectors in the code')

            if verbose:
                print('')
                print(dfinfo(dfr))

            if normalize:
                print('')
                dfr = dfnormalize(dfr, None, verbose)

        # for safety:
        dfr.reset_index(drop=True, inplace=True)
        return dfr

    @classmethod
    def _open(cls, dataframe):
        return dataframe


# ###################################
# AVAILABLE DATASETS. IN PRINCIPLE,
# EACH CLASS BELOW MANAGES A HDF FILE IN the 'datasets' dir
# ####################################

class pgapgv(DatasetInfo):
    '''Dataset with pga and pgv calculated.
    Created from the Europe database (local earthquakes)
    See sod.stream2segment.configs
    (pgapgv.py and pgapgv.yaml)
    '''

    _MODIFIED_COL = 'modified'

    # list of unique columns identifying an instance in this dataset
    # OUTLIER_COL MUST be always present. Also:
    # The first column will replace S2S_COL and will uniquely identify the
    # DataFrame's dataset: it will always be the DataFrame FIRST column
    # (see open_dataset). Set it to this class name + '.id'
    _uid_columns = (_MODIFIED_COL,)

    # list of this dataset (sub)classes:
    classnames = (
        'ok',
        'outl. (wrong inv. file)',
        'outl. (cha. resp. acc <-> vel)',
        'outl. (gain X100 or X0.01)',
        'outl. (gain X10 or X0.1)',
        'outl. (gain X2 or X0.5)'
    )

    # dict where each dataset's subclass is mapped to a selector function:
    # "func(dataframe)" returning the pandas Series of booleans indicating
    # where the dataframe row match the given class, so that, to filter
    # the dataframe with class rows only you call: dataframe[func(dataframe)]
    class_selector = {
        classnames[0]: lambda dataframe: ~is_outlier(dataframe),
        classnames[1]: lambda dataframe:
            dataframe[pgapgv._MODIFIED_COL].str.contains('INVFILE:'),
        classnames[2]: lambda dataframe:
            dataframe[pgapgv._MODIFIED_COL].str.contains('CHARESP:'),
        classnames[3]: lambda dataframe:
            dataframe[pgapgv._MODIFIED_COL].str.contains('STAGEGAIN:X100.0') |
            dataframe[pgapgv._MODIFIED_COL].str.contains('STAGEGAIN:X0.01'),
        classnames[4]: lambda dataframe:
            dataframe[pgapgv._MODIFIED_COL].str.contains('STAGEGAIN:X10.0') |
            dataframe[pgapgv._MODIFIED_COL].str.contains('STAGEGAIN:X0.1'),
        classnames[5]: lambda dataframe:
            dataframe[pgapgv._MODIFIED_COL].str.contains('STAGEGAIN:X2.0') |
            dataframe[pgapgv._MODIFIED_COL].str.contains('STAGEGAIN:X0.5')
    }

    # dict where each dataset's class is mapped to its weight. The weight is
    # only used in html evaluation reports to dynamically sort conf.matrices
    class_weight = {
        classnames[0]: 100,
        classnames[1]: 100,
        classnames[2]: 10,
        classnames[3]: 50,
        classnames[4]: 5,
        classnames[5]: 1
    }

    @classmethod
    def _open(cls, dataframe):
        '''Custom operations to be performed on this dataset'''
        modified_col = cls._MODIFIED_COL
        # setting up columns:
        dataframe['pga'] = np.log10(dataframe['pga_observed'].abs())
        dataframe['pgv'] = np.log10(dataframe['pgv_observed'].abs())
        dataframe['delta_pga'] = np.log10(dataframe['pga_observed'].abs()) - \
            np.log10(dataframe['pga_predicted'].abs())
        dataframe['delta_pgv'] = np.log10(dataframe['pgv_observed'].abs()) - \
            np.log10(dataframe['pgv_predicted'].abs())
        del dataframe['pga_observed']
        del dataframe['pga_predicted']
        del dataframe['pgv_observed']
        del dataframe['pgv_predicted']
        for col in dataframe.columns:
            if col.startswith('amp@'):
                # go to db. We should multuply log * 20 (amp spec) or * 10 (pow
                # spec) but it's unnecessary as we will normalize few lines below
                dataframe[col] = np.log10(dataframe[col])
        # save space:
        dataframe[modified_col] = dataframe[modified_col].astype('category')
        # numpy int64 for just zeros and ones is waste of space: use bools
        # (int8). But first, let's be paranoid first (check later, see below)
        _zum = dataframe[OUTLIER_COL].sum()
        # convert:
        dataframe[OUTLIER_COL] = dataframe[OUTLIER_COL].astype(bool)
        # check:
        if dataframe[OUTLIER_COL].sum() != _zum:
            raise ValueError('The column "outlier" is supposed to be '
                             'populated with zeros or ones, but conversion '
                             'to boolean failed. Check the column')
        return dataframe


class oneminutewindows(DatasetInfo):
    '''Dataset with only noise-related features (psd) <- I GUESS.
    Created from the Europe database (local earthquakes)
    See sod.stream2segment.configs
    (oneminutewindows.py and oneminutewindows.yaml)
    '''

    _WINDOW_TYPE_COL = 'window_type'  # defined in oneminiutewindows.hdf

    # list of unique columns identifying an instance in this dataset
    # OUTLIER_COL MUST be always present. Also:
    # The first column will replace S2S_COL and will uniquely identify the
    # DataFrame's dataset: it will always be the DataFrame FIRST column
    # (see open_dataset). Set it to this class name + '.id'
    _uid_columns = (pgapgv._MODIFIED_COL, _WINDOW_TYPE_COL)

    # list of dataset (sub)classes:
    classnames = pgapgv.classnames

    # see pgapgv
    class_selector = pgapgv.class_selector

    # see pgapgv
    class_weight = pgapgv.class_weight

    @classmethod
    def _open(cls, dataframe):
        '''Custom operations to be performed on this dataset'''
        modified_col = pgapgv._MODIFIED_COL
        # save space:
        dataframe[modified_col] = dataframe[modified_col].astype('category')
        dataframe[cls._WINDOW_TYPE_COL] = \
            dataframe[cls._WINDOW_TYPE_COL].astype('category')
        return dataframe


class oneminutewindows_sn_only(oneminutewindows):
    '''Purged version of (subsey of) oineminutewindows:
    the Europe database (local earthquakes) was divided into three
    subwindows: 1 min noise, 1 min noise+signal, 1min signal. The
    '1min noise+signal' segments were removed here, as practically equivalent
    to 1min signal
    '''
    pass


class magnitudeenergy(DatasetInfo):
    '''Dataset with only noise-related features (psd) <- I GUESS.
    Created from the Me database (teleseismic earthquakes)
    See sod.stream2segment.configs
    (magnitudeenergy.py and magnitudeenergy.yaml)
    '''


    _SUBCLASS_COL = 'subclass'  # defined in magnitudeenergy.hdf

    # list of unique columns identifying an instance in this dataset
    # OUTLIER_COL MUST be always present. Also:
    # The first column will replace S2S_COL and will uniquely identify the
    # DataFrame's dataset: it will always be the DataFrame FIRST column
    # (see open_dataset). Set it to this class name + '.id'
    _uid_columns = (_SUBCLASS_COL,)

    classnames = (
        'ok',
        'outlier',
        'unlabeled (suspicious outl.)',
        'unlabeled (unknown)'
    )

    # dict where each dataset's subclass is mapped to a selector function:
    # "func(dataframe)" returning the pandas Series of booleans indicating
    # where the dataframe row match the given class, so that, to filter
    # the dataframe with class rows only you call: dataframe[func(dataframe)]
    class_selector = {
        classnames[0]: lambda dataframe:
            ~is_outlier(dataframe) &
            dataframe[magnitudeenergy._SUBCLASS_COL].str.match('^$'),
        classnames[1]: lambda dataframe:
            is_outlier(dataframe) &
            dataframe[magnitudeenergy._SUBCLASS_COL].str.match('^$'),
        classnames[2]: lambda df:
            df[magnitudeenergy._SUBCLASS_COL].
            str.contains('unlabeled.maybe.outlier'),
        classnames[3]: lambda df:
            df[magnitudeenergy._SUBCLASS_COL].
            str.contains('unlabeled.unknown')
    }

    # dict where each dataset's class is mapped to its weight. The weight is
    # only used in html evaluation reports to dynamically sort conf.matrices
    class_weight = {
        classnames[0]: 100,
        classnames[1]: 100,
        classnames[2]: 10,
        classnames[3]: 1
    }

    @classmethod
    def _open(cls, dataframe):
        '''Custom operations to be performed on this dataset'''
        subclass_col = cls._SUBCLASS_COL
        # save space:
        dataframe[subclass_col] = dataframe[subclass_col].astype('category')
        # set the outlier where suspect is True as True:
        dataframe.loc[
            dataframe[subclass_col].str.contains('unlabeled.maybe.outlier'),
            OUTLIER_COL] = True
        # dataframe['modified'] = dataframe['modified'].astype('category')
        return dataframe


class globalset(DatasetInfo):
    '''This dataset represents the dataframe created in
    'Create.global.dataset.ipynb' by merging oneminutewindows_sn_only.hdf and
    magnitudeenergy.hdf
    '''
    _SUBCLASS_COL = \
        magnitudeenergy._SUBCLASS_COL  # pylint: disable=protected-access
    _WINDOW_TYPE_COL = \
        oneminutewindows._WINDOW_TYPE_COL  # pylint: disable=protected-access

    # list of unique columns necessary to identify a dataframe under this
    # dataset. These columns must also be passed to any DataFrame generated
    # splitting/filtering/evaluating (see e.g. 'predict' child dataframe
    # ing an instance in this dataset
    # OUTLIER_COL MUST be always present. Also:
    # The first column will replace S2S_COL and will uniquely identify the
    # DataFrame's dataset: it will always be the DataFrame FIRST column
    # (see open_dataset). Set it to this class name + '.id'
    _uid_columns = ('dataset_id', _SUBCLASS_COL, _WINDOW_TYPE_COL)

    classnames = (
        'ok',  # inlier of omw or unknowns me
        'outl. (wrong inv)',  # artificially created in omw or labelled in me
        'outl. (cha. resp. acc <-> vel)',
        'outl. (gain X100 or X0.01)',
        'outl. (gain X10 or X0.1)',
        'outl. (gain X2 or X0.5)',
        'unlabeled (Me suspicious outl.)',
        'unlabeled (Me unknown)'
    )

    # dict where each dataset's subclass is mapped to a selector function:
    # "func(dataframe)" returning the pandas Series of booleans indicating
    # where the dataframe row match the given class, so that, to filter
    # the dataframe with class rows only you call: dataframe[func(dataframe)]
    class_selector = {
        classnames[0]: lambda dataframe:
            ~is_outlier(dataframe) &
            dataframe[globalset._SUBCLASS_COL].str.match('^$'),
        classnames[1]: lambda dataframe:
            is_outlier(dataframe) &
            (dataframe[globalset._SUBCLASS_COL].str.contains('INVFILE:') |
             dataframe[globalset._SUBCLASS_COL].str.match('^$')),
        classnames[2]: lambda dataframe:
            dataframe[globalset._SUBCLASS_COL].str.contains('CHARESP:'),
        classnames[3]: lambda dataframe:
            dataframe[globalset._SUBCLASS_COL].str.contains('STAGEGAIN:X100.0') |
            dataframe[globalset._SUBCLASS_COL].str.contains('STAGEGAIN:X0.01'),
        classnames[4]: lambda dataframe:
            dataframe[globalset._SUBCLASS_COL].str.contains('STAGEGAIN:X10.0') |
            dataframe[globalset._SUBCLASS_COL].str.contains('STAGEGAIN:X0.1'),
        classnames[5]: lambda dataframe:
            dataframe[globalset._SUBCLASS_COL].str.contains('STAGEGAIN:X2.0') |
            dataframe[globalset._SUBCLASS_COL].str.contains('STAGEGAIN:X0.5'),
        classnames[6]: lambda df:
            df[globalset._SUBCLASS_COL].str.contains('unlabeled.maybe.outlier'),
        classnames[7]: lambda df:
            df[globalset._SUBCLASS_COL].str.contains('unlabeled.unknown')
    }

    # dict where each dataset's class is mapped to its weight. The weight is
    # only used in html evaluation reports to dynamically sort conf.matrices
    class_weight = {
        classnames[0]: 100,
        classnames[1]: 100,
        classnames[2]: 10,
        classnames[3]: 50,
        classnames[4]: 5,
        classnames[5]: 1,
        classnames[6]: 1,
        classnames[7]: 1
    }

    @classmethod
    def _open(cls, dataframe):
        '''Custom operations to be performed on this dataset'''
        # save space:
        dataframe[cls._SUBCLASS_COL] = \
            dataframe[cls._SUBCLASS_COL].astype('category')
        dataframe[cls._WINDOW_TYPE_COL] = \
            dataframe[cls._WINDOW_TYPE_COL].astype('category')
        return dataframe


class globalset_inliers(globalset):
    '''
    globalset.hdf with inliers only. Created via the notebook:
    Creating.allset_and_globalset.inliers.noinliers.datasets
    '''
    pass


class globalset_noinliers(globalset):
    '''globalset.hdf with not inliers. This dataset includes all segments not
    necessarily 100% inliers, i.e., it can contain segments labelled with
    outlier=False. Created via the notebook:
    Creating.allset_and_globalset.inliers.noinliers.datasets
    '''
    pass


class allset(DatasetInfo):
    '''This dataset represents the dataframe created in
    'Creating.allset.ipynb' by merging globalset.hdf and cx_chile.hdf
    It also removes all segments synthetically created in order to
    speed up cv testing. These instances can be run later if you want by
    evaluating the dataset 'allset_synth_modified_segments.hdf'
    '''
    _SUBCLASS_COL = \
        magnitudeenergy._SUBCLASS_COL  # pylint: disable=protected-access
    _WINDOW_TYPE_COL = \
        oneminutewindows._WINDOW_TYPE_COL  # pylint: disable=protected-access
    _LOC_COL = 'location_code'
    _CHA_COL = 'channel_code'
    _STAID_COL = 'station_id'

    # list of unique columns identifying an instance in this dataset
    # OUTLIER_COL MUST be always present. Also:
    # The first column will replace S2S_COL and will uniquely identify the
    # DataFrame's dataset: it will always be the DataFrame FIRST column
    # (see open_dataset). Set it to this class name + '.id'
    _uid_columns = ('dataset_id', _SUBCLASS_COL, _WINDOW_TYPE_COL, _CHA_COL,
                   _LOC_COL, _STAID_COL)

    classnames = globalset.classnames[:2] + globalset.classnames[-2:]

    # dict where each dataset's subclass is mapped to a selector function:
    # "func(dataframe)" returning the pandas Series of booleans indicating
    # where the dataframe row match the given class, so that, to filter
    # the dataframe with class rows only you call: dataframe[func(dataframe)]
    class_selector = {_: globalset.class_selector[_] for _ in classnames}

    # dict where each dataset's class is mapped to its weight. The weight is
    # only used in html evaluation reports to dynamically sort conf.matrices
    class_weight = {_: globalset.class_weight[_] for _ in classnames}

    @classmethod
    def _open(cls, dataframe):
        '''Custom operations to be performed on this dataset'''
        # save space:
        for cl_ in [cls._CHA_COL, cls._LOC_COL, cls._SUBCLASS_COL,
                    cls._WINDOW_TYPE_COL]:
            dataframe[cl_] = dataframe[cl_].astype('category')
        return dataframe


class allset_synth_modified_segments(allset):
    '''This dataset represents the dataframe created in
    'Creating.allset.ipynb' by merging globalset.hdf and cx_chile.hdf
    and includes ONLY segments synthetically created in order to
    speed up cv testing.
    '''
    classnames = globalset.classnames[2:-2]


class allset_inliers(allset):
    '''
    allset.hdf with inliers only. Created via the notebook:
    Creating.allset_and_globalset.inliers.noinliers.datasets
    '''
    pass


class allset_noinliers(allset):
    '''
    allset.hdf with not inliers. This dataset includes all segments not
    necessarily 100% inliers, i.e., it can contain segments labelled with
    outlier=False. Created via the notebook:
    Creating.allset_and_globalset.inliers.noinliers.datasets
    '''
    pass


class allset_train(allset):
    '''
    allset.hdf with inliers only for novelty detection training.
    It is purged from inliers
    with artifacts (moved in allset_test). Created via the notebook:
    Creating.dataset.allset_train.allset_test
    '''
    classnames = allset.classnames[:1]


class allset_test(allset):
    '''
    allset.hdf for novelty detection testing.
    This dataset includes also inliers moved from allset_train here, by
    iterating over each station, fetching the station's segments
    where all psd* features are in the
    [0.1, 0.9] quantiles, and then moving 20% of them here (random sampling).
    It also includes inliers which were
    labelled as outliers after inspection (window_type='a').
    Created via the notebook:
    Creating.dataset.allset_train.allset_test
    '''
    classnames = globalset.classnames[:2] + globalset.classnames[-1:]

    # dict where each dataset's subclass is mapped to a selector function:
    # "func(dataframe)" returning the pandas Series of booleans indicating
    # where the dataframe row match the given class, so that, to filter
    # the dataframe with class rows only you call: dataframe[func(dataframe)]
    class_selector = {
        classnames[0]: allset.class_selector[classnames[0]],
        classnames[1]: allset.class_selector[classnames[1]],
        classnames[2]: lambda df:
            df[allset._SUBCLASS_COL].str.contains('unlabeled.')
    }

class europe(allset):
    '''
    europe.hdf for novelty detection testing.
    This dataset includes all non processed segment from the Europe dataset (s2s_2019)
    which have data_seed_id not null (see stream2segment). All segments are
    (theoretically) inliers.
    '''

    classnames = ['sd2_2019 unknown']

    class_selector = {
        classnames[0]: lambda dataframe:
            ~is_outlier(dataframe) &
            dataframe[globalset._SUBCLASS_COL].str.match('^$'),
    }
###########################
# Other operations
###########################


@contextmanager
def capture_stderr(verbose=False):
    '''Context manager to be used in a with statement in order to capture
    std.error messages (e.g., python warnings):
    ```
    with capture_stderr():
        ... code here
    ```
    :param verbose: boolean (default False). If True, prints the captured
        messages (if present)
    '''
    # Code to acquire resource, e.g.:
    # capture warnings which are redirected to stderr:
    syserr = sys.stderr
    if isinstance(syserr, StringIO):
        # already within a captured_stderr with statement?
        yield
    else:
        captured_err = StringIO()
        sys.stderr = captured_err
        try:
            yield
            if verbose:
                errs = captured_err.getvalue()
                if errs:
                    print('')
                    print('During the operation, '
                          'the following warning(s) were issued:')
                    print(errs)
            captured_err.close()
        finally:
            # restore standard error:
            sys.stderr = syserr


def dfinfo(dataframe, asstring=True):
    '''Returns a a dataframe with info about the given `dataframe` representing
    a given dataset
    '''
    dinfo = dataset_info(dataframe)
    classes = {c: dinfo.class_selector[c] for c in dinfo.classnames}
    sum_dfs = odict()
    empty_classes = set()
    infocols = ['Min', 'Median', 'Max', '#NAs', '#<1Perc.', '#>99Perc.']
    for classname, class_selector in classes.items():
        sum_df = odict()
        _dfr = dataframe[class_selector(dataframe)]
        if _dfr.empty:
            empty_classes.add(classname)
            continue
        # if _dfr.empty
        for col in floatingcols(dataframe):
            q01 = np.nanquantile(_dfr[col], 0.01)
            q99 = np.nanquantile(_dfr[col], 0.99)
            df1, df99 = _dfr[(_dfr[col] < q01)], _dfr[(_dfr[col] > q99)]
            # segs1 = len(pd.unique(df1[ID_COL]))
            # segs99 = len(pd.unique(df99[ID_COL]))
            # stas1 = len(pd.unique(df1['station_id']))
            # stas99 = len(pd.unique(df99['station_id']))

            sum_df[col] = {
                infocols[0]: np.nanmin(_dfr[col]),
                infocols[1]: np.nanquantile(_dfr[col], 0.5),
                infocols[2]: np.nanmax(_dfr[col]),
                infocols[3]: (~np.isfinite(_dfr[col])).sum(),
                infocols[4]: len(df1),
                infocols[5]: len(df99)
                # columns[5]: stas1 + stas99,
            }
        sum_dfs[classname + " (%d instances)" % len(_dfr)] = \
            pd.DataFrame(data=list(sum_df.values()),
                         columns=infocols,
                         index=list(sum_df.keys()))

#     return df2str(pd.DataFrame(data, columns=columns, index=index))
    if not asstring:
        return pd.concat(sum_dfs.values(), axis=0, keys=sum_dfs.keys(),
                         sort=False)

    allstrs = []
    for (key, val) in sum_dfs.items():
        allstrs.extend(['', key, val.to_string()])
    return '\n'.join(allstrs)


# def df2str(dataframe):
#     ''':return: the string representation of `dataframe`, with numeric values
#     formatted with comma as decimal separator
#     '''
#     return _dfformat(dataframe).to_string()


def _dfformat(dataframe, n_decimals=2):
    '''Returns a copy of `dataframe` with all numeric values converted to
    formatted strings (with comma as thousand separator)

    :param n_decimals: how many decimals to display for floats (defautls to 2)
    '''
    float_frmt = '{:,.' + str(n_decimals) + 'f}'  # e.g.: '{:,.2f}'
    strformat = {
        c: "{:,d}" if str(dataframe[c].dtype).startswith('int') else float_frmt
        for c in dataframe.columns
    }
    return pd.DataFrame({c: dataframe[c].map(strformat[c].format)
                         for c in dataframe.columns},
                        index=dataframe.index)


def dfnormalize(dataframe, columns=None, verbose=True):
    '''Normalizes dataframe under the sepcified columns. Only good instances
    (not outliers) will be considered in the normalization

    :param columns: if None (the default), nornmalizes on floating columns
        only. Otherwise, it is a list of strings denoting the columns on
        which to normalize
    '''
    if verbose:
        if columns is None:
            print('Normalizing numeric columns (floats only)')
        else:
            print('Normalizing %s' % str(columns))
        print('Normalization is a Rescaling (min-max normalization) where '
              'mina and max are calculated on inliers only'
              'and applied to all instances)')

    with capture_stderr(verbose):
        norm_df = dataframe[~is_outlier(dataframe)]
        itercols = floatingcols(dataframe) if columns is None else columns
        for col in itercols:
            # for calculating min and max, we need to drop also infinity, tgus
            # np.nanmin and np.nanmax do not work. Hence:
            finite_values = norm_df[col][np.isfinite(norm_df[col])]
            min_, max_ = np.min(finite_values), np.max(finite_values)
            dataframe[col] = (dataframe[col] - min_) / (max_ - min_)
        if verbose:
            print(dfinfo(dataframe))

    return dataframe


def floatingcols(dataframe):
    '''Iterable yielding all floating point columns of dataframe'''
    for col in dataframe.columns:
        try:
            if np.issubdtype(dataframe[col].dtype, np.floating):
                yield col
        except TypeError:
            # categorical data falls here
            continue


####################
# TO BE TESTED!!!!
####################


NUM_SEGMENTS_COL = 'num_segments'


def is_station_df(dataframe):
    '''Returns whether the given dataframe is the result of `groupby_station`
    on a given segment-based dataframe
    '''
    return NUM_SEGMENTS_COL in dataframe.columns


def groupby_stations(dataframe, verbose=True):
    '''Groups `dataframe` by stations and returns the resulting dataframe
    Numeric columns are merged taking the median of all rows
    '''
    if verbose:
        print('Grouping dataset per station')
        print('(For floating columns, the median of all segments stations '
              'will be set)')
        print('')
    with capture_stderr(verbose):
        newdf = []
        fl_cols = list(floatingcols(dataframe))
        for (staid, modified, outlier), _df in \
                dataframe.groupby(['station_id', 'modified', 'outlier']):
            _dfmedian = _df[fl_cols].median(axis=0, numeric_only=True,
                                            skipna=True)
            _dfmedian[NUM_SEGMENTS_COL] = len(_df)
            _dfmedian['outlier'] = outlier
            _dfmedian['modified'] = modified
            _dfmedian[ID_COL] = staid
            newdf.append(pd.DataFrame([_dfmedian]))
            # print(pd.DataFrame([_dfmedian]))

        ret = pdconcat(newdf, ignore_index=True)
        ret[NUM_SEGMENTS_COL] = ret[NUM_SEGMENTS_COL].astype(int)
        # convert dtypes because they might not match:
        shared_c = (set(dataframe.columns) & set(ret.columns)) - set(fl_cols)
        for col in shared_c:
            ret[col] = ret[col].astype(dataframe[col].dtype)
        if verbose:
            bins = [1, 10, 100, 1000, 10000]
            max_num_segs = ret[NUM_SEGMENTS_COL].max()
            if max_num_segs >= 10 * bins[-1]:
                bins.append(max_num_segs + 1)
            elif max_num_segs >= bins[-1]:
                bins[-1] = max_num_segs + 1
            groups = ret.groupby(pd.cut(ret[NUM_SEGMENTS_COL], bins,
                                        precision=0,
                                        right=False))
            print(pd.DataFrame(groups.size(), columns=['num_stations']).
                  to_string())
            assert groups.size().sum() == len(ret)
            print('')
            print('Summary of the new dataset (instances = stations)')
            print(dfinfo(ret))
        return ret
