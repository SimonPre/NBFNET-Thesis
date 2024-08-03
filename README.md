# README for: A closer look at Neural Bellman-Ford networks

## Installation and Set-Up

### Database Access 
In order to use the provided codebase please set up a Project in google Firebase and create
a Firestore database. For details please see (https://firebase.google.com/docs/firestore/quickstart)
This is the only form of supported result storage.
After setting up the database, please create a  service account using the cloud console and download
an access token as yaml. Next you need to create a .env file in the `/util/db` subfolder of this codebase.
In this `.env` file please provide `key=<your string>` where your string is the string representation of the created access key.
To obtain a valid string representation you can use the yaml library of Python to convert the downloaded access
token to a string.

### Installation.

For installation an environment.yml file is provided.
In case you encounter "dnn != nullptr", please update your cudnn installation.
For an exact replication the file
environment_droplet.yml contains an exact description of the conda environment used for
the thesis.

## Running the model.

The model can currently only be run on GPU. 
To run the model please run the search.py file Unfortunately I did not have enough time to create the
possibility for command line options. Therefore, some fine-tuning needs to be done in the main method
in search.py

Three cases can be distinguished. 
### Running the base model
If you wish to run the model defined in the base base_conf.yaml file. Simply use:

    if __name__ == "__main__":
        Search(n_trials=1, search_type="search", devices="3").search(debug=True)"

This executes the model as defined in the base_config file, the search_type parameter
is ignored

### Running multiple pre-defined combinations
If you intend to run multiple training runs, the easiest approach is to navigate to the
folder search/search_space and create a folder called ablation. Copy the base_conf.yaml as many times
as needed and make the changes you would like to make. For possible options and the meaning of all
parameters please consult the base_conf.yaml. 

Change the main method in search.py to:

    if __name__ == "__main__":
        Search(n_trials=10, search_type="ablation", devices="2,6").search(debug=False)"

In this example we want to run 10 different configurations, using the gpus 2 and 6 and the server.
The debug parameter needs to be set to false, and the search_type needs to be specified as ablation.
The code will train all the supplied configurations and automatically distribute the training runs,
between the supplied gpus. 

### Running a search

This scenario is the most limited. If you want to run a search, use libkge to generate the trials.
Copy the entire trial folders into the search/search_space folder, and make sure that any sub-folder
called ablation has been deleted beforehand. Run: 

    if __name__ == "__main__":
        Search(n_trials=10, search_type="search", devices="2,6").search(debug=False)"

Please note that only the parameters presented in the thesis are supported. Furthermore, the file
search_space_generation_libkge.yaml in the util/config folder provides the configuration used to create the used hyperparamters searches. Please do not 
change the names of the parameters if you want to run a similar search. Again the model
takes care of the distribution of jobs to GPUs.

## Adding a new dataset

To add a new dataset, make sure it follows the libkge conventions and just copy the folder
to the datasets folder in the data folder. 

## Replicating the results produced in my Thesis

To replicate the results, please create a config in which all hyperparamters are set to the reported defaults.
And modify the setting as reported in the thesis. Running this configuration should lead to the same results +- model
variance. 

The file analysis/all_searches.csv provides all runs of model training which have produced results on test.
The analysing_searches jupyter notebook contains the analysis underlying most tables. With the exception
of the last experiments regarding the indicator function, as these were directly taken from the database. 

    




