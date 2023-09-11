# creates a file with the contents of this cell to file

from azureml.core import Run
import pandas as pd
import os

run = Run.get_context()
# For example, when running a script as part of an AzureML experiment,
# this context contains information about the experiment run, such as 
# the run ID, the workspace, and other relevant details. -> ChatGPT explanation
run.clean
data = pd.read_csv('diabetes.csv')

# counting rows and logging result
row_count = len(data)
run.log('Observations:',row_count)
print(f'Analyzing {row_count} rows of data.')

diabetic_counts = data['Diabetic'].value_counts()
print(f'Diabetic Counts:')
for key, value in diabetic_counts.items():
    run.log('Label:' + str(key), str(value))

# Save a sample of data input to the outputs folder
os.makedirs('outputs',exist_ok = True) # create the directory first
data.sample(100).to_csv('outputs/sample.csv',index=False,header=True)

run.complete()
