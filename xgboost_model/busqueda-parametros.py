import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn
from hyperopt import hp, tpe, fmin
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer

seed = 798589991

# Dividimos entre la data que tenemos y la de evaluación para submitear.
comp_data = pd.read_csv("data/eng_data.csv")

# La información que tenemos para entrenar y validar.
local_data = comp_data[comp_data["ROW_ID"].isna()]

# La información en la que no tenemos las y, para predecir con el modelo ya entrenado y subir a Kaggle.
kaggle_data = comp_data[comp_data["ROW_ID"].notna()] 

# Entrenamos un modelo de xgboost
y = local_data[['conversion']].copy()
X = local_data.drop(columns=['conversion', 'ROW_ID'], axis = 1)

val_test_size = 0.3 # Proporción de la suma del test de validación y del de test.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = val_test_size, random_state = seed, stratify = y)

# Define the search space for hyperparameters
space = {
    'max_depth': hp.choice('max_depth', range(1, 10)),
    'learning_rate': hp.loguniform('learning_rate', -5, 0),
    'n_estimators': hp.choice('n_estimators', range(50, 300)),
    'gamma': hp.loguniform('gamma', -5, 0),
    'subsample': hp.uniform('subsample', 0.5, 1.0),  # Add subsample
    'min_child_weight': hp.choice('min_child_weight', range(1, 10)),  # Add min_child_weight
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),  # Add colsample_bytree
    'reg_lambda': hp.loguniform('reg_lambda', -5, 5),  # Add reg_lambda
}

# Objective function for hyperparameter optimization
def objective(params):
    cls =  make_pipeline(StandardScaler(), SimpleImputer(), xgb.XGBClassifier(
        objective='binary:logistic',
        seed=seed,
        eval_metric='auc',
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        gamma=params['gamma'],
        subsample=params['subsample'],  # Use subsample
        min_child_weight=params['min_child_weight'],  # Use min_child_weight
        colsample_bytree=params['colsample_bytree'],  # Use colsample_bytree
        reg_lambda=params['reg_lambda'],  # Use reg_lambda
    )
)

    cls.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    y_pred = cls.predict_proba(X_val)[:, 1]
    auc_roc = sklearn.metrics.roc_auc_score(y_val, y_pred)
    
    return -auc_roc  # We want to maximize AUC-ROC, so we negate it for minimization

# Set up Hyperopt search
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=350)  # Adjust max_evals as needed

# Print the best hyperparameters
print("Best Hyperparameters:")
print(best)