import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn
from hyperopt import hp, tpe, fmin
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import gensim
import re
import nltk
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer, confusion_matrix
from sklearn.preprocessing import StandardScaler

seed = 798589991

# Para W2V
def tokenizer(raw_text):
    """
    Tokeniza y preprocesa un texto.

    Args:
        raw_text (str): Texto sin procesar.

    Returns:
        list: Lista de oraciones, donde cada oración es una lista de palabras.
    """
    sentences = sent_tokenize(raw_text)
    sentences = [word_tokenize(e) for e in sentences]
    sentences = [[e2 for e2 in e1 if re.compile("[A-Za-z]").search(e2[0])] for e1 in sentences]
    sentences = [[e2.lower() for e2 in e1] for e1 in sentences]
    return(sentences)

def average_vectors(title_tokens, model, stopwords=None):
    """
    Calcula el vector promedio de un conjunto de tokens utilizando un modelo Word2Vec.

    Args:
        title_tokens (list): Lista de tokens.
        model (gensim.models.Word2Vec): Modelo Word2Vec.
        stopwords (set, optional): Conjunto de palabras stopwords. Defaults to None.

    Returns:
        numpy.ndarray: Vector promedio.
    """
    title_tokens = [e2 for e1 in title_tokens for e2 in e1]
    title_tokens = [e for e in title_tokens if e in model.wv]
    if stopwords is not None:
        title_tokens = [e for e in title_tokens if e not in stopwords]
    if len(title_tokens) == 0:
        output = np.zeros(model.wv.vector_size)
    else:
        output = np.array([model.wv.get_vector(e) for e in title_tokens]).mean(0)
    return output

def data_eng(df):
    # Deshacemos la columna platforms en las distintas plataformas de las que puede ver, Ios, Android, Desktop o Mobile (la pagina en un cel, no la app.)
    df['platform'] = df['platform'].str.split('/').str[2]

    # Deshacemos el string de title con flechas en columnas con su categoria global y categoria final.
    df['category_first'] = df['full_name'].str.split(' -> ').str[0]
    df['category_last'] = df['full_name'].str.split(' -> ').str[-1]
    df.drop('full_name', inplace=True, axis=1)

    # Columna del porcentaje de descuento
    discount = (((df['original_price'] - df['price']) / df['original_price']) * 100).astype(int)
    df['discount_%'] = discount
    df.drop('original_price', inplace=True, axis=1)

    # Columnas derivadas de los tags.
    unique_tags = []
    for list in df['tags']:
        list_split = list[1:len(list)-1].split(', ')
        for item in list_split:
            if not (item in unique_tags):
                unique_tags.append(item)
    for tag in unique_tags:
        df[tag] = df['tags'].apply(lambda x: tag in x)
    df.drop('tags', inplace=True, axis=1)

    # Columnas derivadas de la fecha y horario de la sesión.
    df['date'] = pd.to_datetime(df['print_server_timestamp'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['second'] = df['date'].dt.second
    df.drop('date', inplace=True, axis=1)
    df.drop('print_server_timestamp', inplace=True, axis=1)

    # Columna garantía
    warranty_types = df['warranty']
    warranty_types = warranty_types.reset_index()
    warranty_types['warranty_words'] = warranty_types['warranty'].apply(lambda x: str(x).upper().split() if not pd.isna(x) else [])

    words = ['SIN','GARANTÍA','GARANTIA']

    for word in words:
        warranty_types[word] = warranty_types['warranty_words'].apply(lambda x: word in x)

    warranty_types['Numbers'] = warranty_types['warranty'].str.extract(r'(\d+)')
    warranty_types['has_warranty'] = ~(((warranty_types['SIN'] & warranty_types['GARANTÍA']) | (warranty_types['SIN'] & warranty_types['GARANTIA'])) & warranty_types['Numbers'].isna())

    df['warranty'] = warranty_types['has_warranty']
    
    # Dropeo columnas innecesarias
    df.drop('benefit', inplace=True, axis=1)
    df.drop('user_id', inplace=True, axis=1)
    df.drop('uid', inplace=True, axis=1)
    df.drop('main_picture', inplace=True, axis=1)
    df.drop('category_id', inplace=True, axis=1)
    df.drop('domain_id', inplace=True, axis=1)
    df.drop('deal_print_id', inplace=True, axis=1)
    df.drop('etl_version', inplace=True, axis=1)
    df.drop('product_id', inplace=True, axis=1)
    df.drop('site_id', inplace=True, axis=1)
    df.drop('item_id', inplace=True, axis=1)
    df.drop('accepts_mercadopago', inplace=True, axis=1)

    # OHE de variables categóricas
    cols_to_encode = ['category_first', 'listing_type_id', 'logistic_type', 'platform', 'category_last']
    df_encoded = pd.get_dummies(df[cols_to_encode])
    df = pd.concat([df, df_encoded], axis=1)
    df.drop(columns=cols_to_encode, inplace=True, axis=1)

    df['title_tokens'] = df['title'].map(tokenizer)

    STOP_WORDS_SP = set(stopwords.words('spanish'))

    # Creación del modelo Word2Vec
    vec_size = 50
    w2v_title = gensim.models.Word2Vec(vector_size=vec_size,
                                    window=9,
                                    min_count=5,
                                    negative=15,
                                    sample=0.01,
                                    workers=8,
                                    sg=1)

    # Creación del vocabulario a partir del corpus
    w2v_title.build_vocab([e2 for e1 in df['title_tokens'].values for e2 in e1], 
                        progress_per=10000)

    # Entrenamiento del modelo Word2Vec
    w2v_title.train([e2 for e1 in df['title_tokens'].values for e2 in e1],
                    total_examples=w2v_title.corpus_count,
                    epochs=30, report_delay=1)

    title_embs = df['title_tokens'].map(lambda x: average_vectors(x, w2v_title, STOP_WORDS_SP))
    embedding_title_columns = pd.DataFrame(title_embs.tolist(), columns=[f'title_emb_{i}' for i in range(vec_size)])

    df.drop('title_tokens', inplace=True, axis=1)
    df.drop('title', inplace=True, axis=1)
    df = pd.concat([df, embedding_title_columns], axis=1)

    # Antes de empezar el entrenamiento del modelo, paso a int las columnas de booleano.
    df.replace({False: 0, True: 1}, inplace=True)

    return df


# Ingeniería de atributos que no hace data leakage al darle todo el dataset (OHE pero no es grave, es conveniente hacerlo así para que no haya mismatch de columnas)
comp_data = pd.read_csv("data/competition_data.csv")
comp_data = data_eng(comp_data)

# Dividimos entre la data que tenemos y la de evaluación para submitear.

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
    'max_depth': hp.choice('max_depth', range(1, 15)),
    'learning_rate': hp.loguniform('learning_rate', -5, 0),
    'n_estimators': hp.choice('n_estimators', range(50, 500)),
    'gamma': hp.loguniform('gamma', -5, 0),
    'subsample': hp.uniform('subsample', 0.5, 1.0),  # Add subsample
    'min_child_weight': hp.choice('min_child_weight', range(1, 10)),  # Add min_child_weight
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),  # Add colsample_bytree
    'reg_lambda': hp.loguniform('reg_lambda', -5, 5),  # Add reg_lambda
}

# Objective function for hyperparameter optimization
def objective(params):
    cls =  xgb.XGBClassifier(
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

    cls.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
    
    y_pred = cls.predict_proba(X_val)[:, 1]
    auc_roc = sklearn.metrics.roc_auc_score(y_val, y_pred)
    
    return -auc_roc  # We want to maximize AUC-ROC, so we negate it for minimization

# Set up Hyperopt search
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100)  # Adjust max_evals as needed

# Print the best hyperparameters
print("Best Hyperparameters:")
print(best)

best_max_depth = range(1, 15)[best['max_depth']]
best_learning_rate = best['learning_rate']
best_n_estimators = range(50, 500)[best['n_estimators']]
best_gamma = best['gamma']
best_subsample = best['subsample']
best_min_child_weight = best['min_child_weight']
best_colsample_bytree = best['colsample_bytree']
best_reg_lambda = best['reg_lambda']

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
