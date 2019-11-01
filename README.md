# sod
A machine learning experimental project to detect seismic events outliers


# dataset creation
Move to sod directory
Activate virtualenv
```bash
process -d postgresql://<user>:<pwsd>@<host>/<dbname> -c ./sod/dataset/executions/oneminutewindows.yaml -p ./sod/dataset/executions/oneminutewindows.py ./tmp/datasets/onwminutewindows.hdf
```


# Evaluation results (FIXME: TO be improved):
Move to sod directory
Activate virtualenv
```bash
export PYTHONPATH='.' && python sod/evaluation.execute.py -c config_file
```
