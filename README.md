# sod
A machine learning experimental project to detect seismic events outliers

The sod ROOT dir is the one in which you clone this package (usually called 'sod'.
Thereing you will have a nested 'sod' directory with the python code, and a 'test' directory
where tests are implemented, plus other files, e.g. requirements.txt)


# dataset creation

A dataset is a dataframe (HDF file) with input data for training
and creating a classifier (or testing and already created classifier).
To create a new dataset with name <dataset>:

1. Implement <dataset>.yaml and <dataset>.py in sod/stream2segment/configs
   (for info, see stream2segment documentation)

2. Move to ROOT
   Activate virtualenv
   With a given input database path, execute:
   ```bash
   s2s process -d postgresql://<user>:<pwsd>@<host>/<dbname> -mp -c ./sod/stream2segment/configs/<dataset>.yaml -p ./sod/stream2segment/configs/<dataset>.py ./sod/datasets/<dataset>.hdf
   ```

A new <dataset>.hdf file is created.


# copy files from different repos:

.model and .hdf files are ignored in git because too big in size, so you will need to copy them
with rsync or scp in case, e.g.:

Move locally to ROOT
(with <ROOT>, we denote the ROOT sod directory on the remote computer). Then:

rsync -auv <user>@<host>:<ROOT>/sod/datasets/<dataset>.hdf ./sod/datasets/
scp <user>@<host>:<ROOT>/sod/datasets/<dataset>.hdf ./sod/datasets/


# CV Evaluation results:

Cross validation evaluation take a scikit learn classifier and a sod CVEvaluator
(which must be implemented, see sod/evaluate.py).
After that, we implement a config file (in sod/evaluations/configs), usually starting with
"cv." (preferable) followed by the dataset file NAME (strongly suggested): e.g., "cv.pgapgv.yaml"
In the file, we implement the necessary parameters and the program will train and test
the classifier(s) for any combination of those parameters.
Results are saved in the directory '/sod/evaluation/<configfilename>':
- N model file (classifiers, one for each parameters set)
- N prediction files (hdf files with the predictions of all elements in the input dataset)
- one HTML report with % recognized and log loss (sort of)

To run a CVevaluation:

Move to ROOT
Activate virtualenv
```bash
export PYTHONPATH='.' && python sod/execute.py -c sod/evaluations/configs/<yamlfile>
```

(note that we need to set the PYTHONPATH as sod is NOT installed as python package)


<!--

# copy back data (TO BE DONE):

scp <host>:<sod_directory>/tmp/evaluation-results/pgapgv/*.html ./tmp/evaluation-results/pgapgv

(replace pgapgv with the folder of your evaluation)



rsync -auv <user>@<host>:<ROOT>/sod/evaluations/results/<DIRECTORY>*.html ./sod/evaluations/results/<DIRECTORY>/

scp <user>@<host>:<ROOT>/sod/evaluations/results/<DIRECTORY>*.html ./sod/evaluations/results/<DIRECTORY>/
-->