'''
Just a container for file paths

Created on 13 Nov 2019

@author: riccardo
'''

from os.path import abspath, join, dirname


DATASETS_DIR = abspath(join(dirname(__file__), '..', 'datasets'))


_EVALUATIONS_ROOT = abspath(join(dirname(__file__), '..', 'evaluations'))


EVALUATIONS_CONFIGS_DIR = abspath(join(_EVALUATIONS_ROOT, 'configs'))


EVALUATIONS_RESULTS_DIR = abspath(join(_EVALUATIONS_ROOT, 'results'))
