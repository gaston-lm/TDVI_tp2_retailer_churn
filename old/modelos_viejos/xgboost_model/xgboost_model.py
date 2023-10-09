import pandas as pd
import xgboost as xgb
import sklearn
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

best_max_depth = 6
best_learning_rate = 0.09512855850107411
best_n_estimators = 224
best_gamma = 0.10113552100045467
best_subsample = 0.9158989170972806
best_min_child_weight = 3
best_colsample_bytree = 0.9000335771350197
best_reg_lambda = 12.654060098206006

final_cls = make_pipeline(xgb.XGBClassifier(
        objective='binary:logistic',
        seed=seed,
        eval_metric='auc',
        max_depth=best_max_depth,
        learning_rate=best_learning_rate,
        n_estimators=best_n_estimators,
        gamma=best_gamma,
        subsample=best_subsample,
        min_child_weight=best_min_child_weight,
        colsample_bytree=best_colsample_bytree,
        reg_lambda=best_reg_lambda,
    )
)

final_cls.fit(X_train, y_train)

# Chequeamos el valor debajo de la curva AUC-ROC
y_pred = final_cls.predict_proba(X_val)[:, 1]
auc_roc = sklearn.metrics.roc_auc_score(y_val, y_pred)
print('AUC-ROC validación: %0.5f' % auc_roc)

# Para hacer submit.
all_data_cls = make_pipeline(xgb.XGBClassifier(
    objective='binary:logistic',
    seed=seed,
    eval_metric='auc',
    max_depth=best_max_depth,
    learning_rate=best_learning_rate,
    n_estimators=best_n_estimators,
    gamma=best_gamma,
    subsample=best_subsample,
    min_child_weight=best_min_child_weight,
    colsample_bytree=best_colsample_bytree,
    reg_lambda=best_reg_lambda,
))

all_data_cls.fit(X, y)

# Predicción en la data de kaggle para submitear.
kaggle_data = kaggle_data.drop(columns=["conversion"])
y_preds = all_data_cls.predict_proba(kaggle_data.drop(columns=["ROW_ID"]))[:, final_cls.classes_ == 1].squeeze()

# Generamos el archivo para submit en base a lo predicho.
submission_df = pd.DataFrame({"ROW_ID": kaggle_data["ROW_ID"], "conversion": y_preds})
submission_df["ROW_ID"] = submission_df["ROW_ID"].astype(int)
submission_df.to_csv("xgboost_model/xgboost_model.csv", sep=",", index=False)