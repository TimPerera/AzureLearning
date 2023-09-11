
from azureml.core import Run 
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

run = Run.get_context()
print('run context:\n',run)

print('Loading Data')
data = pd.read_csv('./diabetes.csv')
X, y = data[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, data['Diabetic'].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

# Regularization state
reg = 0.01

# Train Logistic Regression Model
print('Training a logistic regression model with a regularization rate of',reg)
run.log('Regularization Rate',np.float(reg))
model = LogisticRegression(C=1/reg, solver='liblinear').fit(X_train, y_train)

predicted = model.predict(X_test)
accuracy = np.average(predicted==y_test)
print('Accuracy:',accuracy)
run.log('Accuracy',accuracy)

# Calculating AUC
pred_probs = model.predict_proba(X_test)
auc = roc_auc_score(y_test, pred_probs[:,1])
print('AUC:',auc)
run.log('AUC',np.float(auc))

# Saving the trained model 
os.makedirs('output',exist_ok=True)
joblib.dump(value=model, filename='outputs/diabetes_model.pkl')

run.complete()
