import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn
import re
import nltk
from hyperopt import hp, tpe, fmin
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import gensim

seed = 798589991

comp_data = pd.read_csv("data/eng_data.csv")
all_data = pd.read_csv("data/competition_data.csv")

comp_data.drop('accepts_mercadopago', inplace=True, axis=1)
all_data.drop('accepts_mercadopago', inplace=True, axis=1)

title = all_data['title']
comp_data = pd.concat([comp_data, title], axis=1)

comp_data = comp_data.loc[:,~(comp_data.columns.str.startswith("type_product"))]

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

comp_data["title_tokens"] = comp_data["title"].map(tokenizer)

STOP_WORDS_SP = set(stopwords.words('spanish'))

# Creación del modelo Word2Vec
w2v_tp = gensim.models.Word2Vec(vector_size=300,
                                window=3,
                                min_count=5,
                                negative=15,
                                sample=0.01,
                                workers=8,
                                sg=1)

# Creación del vocabulario a partir del corpus
w2v_tp.build_vocab([e2 for e1 in comp_data["title_tokens"].values for e2 in e1],
                   progress_per=10000)

# Entrenamiento del modelo Word2Vec
w2v_tp.train([e2 for e1 in comp_data["title_tokens"].values for e2 in e1],
             total_examples=w2v_tp.corpus_count,
             epochs=30, report_delay=1)

title_embs = comp_data["title_tokens"].map(lambda x: average_vectors(x, w2v_tp, STOP_WORDS_SP))
embedings = title_embs.to_list()

comp_data['embedings'] = embedings
comp_data.drop('title', inplace=True, axis=1)

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

best_max_depth = 6
best_learning_rate = 0.09512855850107411
best_n_estimators = 224
best_gamma = 0.10113552100045467
best_subsample = 0.9158989170972806
best_min_child_weight = 3
best_colsample_bytree = 0.9000335771350197
best_reg_lambda = 12.654060098206006

final_cls = make_pipeline(StandardScaler(), SimpleImputer(), xgb.XGBClassifier(
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