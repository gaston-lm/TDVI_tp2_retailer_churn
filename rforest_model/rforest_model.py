import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

seed = 798589991

# Cargamos la data de la competencia.
comp_data = pd.read_csv("data/competition_data.csv")

# Dividimos entre la data que tenemos y la de evaluación para submitear.

# La información que tenemos para entrenar y validar.
local_data = comp_data[comp_data["ROW_ID"].isna()]

# La información en la que no tenemos las y, para predecir con el modelo ya entrenado y subir a Kaggle.
kaggle_data = comp_data[comp_data["ROW_ID"].notna()] 

# Entrenamos un modelo de random forest.
y = local_data[['conversion']].copy()
X = local_data.drop(columns=['conversion', 'ROW_ID', 'category_id'], axis = 1)

val_test_size = 0.3 # Proporción de la suma del test de validación y del de test.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = val_test_size, random_state = seed, stratify = y)

cls = make_pipeline(SimpleImputer(), RandomForestClassifier(max_depth=8, random_state=7589))
cls.fit(X_train, y_train)

# Predicción en la data de kaggle para submitear.
kaggle_data = kaggle_data.drop(columns=["conversion"])
kaggle_data = kaggle_data.select_dtypes(include='number')
y_preds = cls.predict_proba(kaggle_data.drop(columns=["ROW_ID"]))[:, cls.classes_ == 1].squeeze()

# Generamos el archivo para submit en base a lo predicho.
submission_df = pd.DataFrame({"ROW_ID": kaggle_data["ROW_ID"], "conversion": y_preds})
submission_df["ROW_ID"] = submission_df["ROW_ID"].astype(int)
submission_df.to_csv("rforest_model/rforest_model.csv", sep=",", index=False)
