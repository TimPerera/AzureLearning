
import argparse
import azureml.core
from azureml.core import Run
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--regularization',type=float, dest='reg_rate', default=0.01, help='regularization rate')
parser.add_argument('--input-data',type=str, dest='training_dataset_id', help='training dataset')
args = parser.parse_args()

reg = args.reg_rate

# Get Experiment run context 
run = Run.get_context()

# Get training dataset
diabetes = run.input_datasets['training_data'].to_pandas_dataframe()

# Seperate features and labels 
X, y = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, diabetes['Diabetic'].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

# Train a logistic model regression
run.log('Regularization Rate',reg)
model = LogisticRegression(C=1/reg, solver='liblinear').fit(X_train,y_train)
y_predicted = model.predict(X_test)

# Check accuracy
accuracy = np.average(y_predicted==y_test)
run.log('Accuracy', accuracy)

# Calculate AUC
pred_probs = model.predict_proba(X_test)
auc = roc_auc_score(y_test, pred_probs[:,1])
run.log('AUC',auc)

os.makedirs('outputs',exist_ok=True)
joblib.dump(value=model, filename='outputs/diabetes_model.pkl')

run.complete()
print('Run Complete')
