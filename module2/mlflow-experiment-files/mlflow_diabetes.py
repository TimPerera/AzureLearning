# Experiment Script

# In an experiment script, ML Flow automatically sets the tracking URI, but we need to include the mlflow packages (next cell)
from azureml.core import Run 
import pandas as pd
import mlflow

with mlflow.start_run():
    data = pd.read_csv('diabetes.csv')
    row_count = len(data)
    mlflow.log_metric('Observations', row_count)
    print('Run Complete')

    

    
