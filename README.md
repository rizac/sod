# sod
A machine learning experimental project to detect seismic events outliers


# dataset creation
Move to sod directory
Activate virtualenv
```bash
process -d postgresql://<user>:<pwsd>@<host>/<dbname> -mp -c ./sod/dataset/executions/oneminutewindows.yaml -p ./sod/dataset/executions/oneminutewindows.py ./tmp/datasets/onwminutewindows.hdf
```

# copy from remote to local:
Move to sod directory
scp <host>:<sod_directory>/tmp/datasets/pgapgv.hdf ./tmp/datasets/pgapgv.hdf

(replace pgapgv.hdf with your hdf file)



# Evaluation results (FIXME: TO be improved):
Move to sod directory
Activate virtualenv
```bash
export PYTHONPATH='.' && python sod/evaluation.execute.py -c config_file
```


# copy back data (TO BE DONE):

scp <host>:<sod_directory>/tmp/evaluation-results/pgapgv/*.html ./tmp/evaluation-results/pgapgv

(replace pgapgv with the folder of your evaluation)