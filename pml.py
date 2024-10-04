#%%

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

#%%
#Funcion 1 para descarga de datos (bank-transformed-train.csv y bank-transformed-test-no-labels.csv)

def load_data(fname):
    try:
        return pd.read_csv(fname, sep=';')
    except Exception as e:
        print(f"Error al cargar el archivo '{fname}': {e}")
        return None
    
        
#%%
#Funcion 2 para transformar los atributos del dataframe

class DataTransformer:
    def __init__(self):
        self.df_entrenamiento = None

    def prepare_data(self, df, test=False):
        self.df_entrenamiento = df.copy()
        self.df_entrenamiento.drop(columns=['pdays', 'contact', 'previous'], inplace=True)
        
        educ_order = {'basic': 0, 'high.school': 1, 'university.degree': 2, 'professional.course': 3, 'other': 4}
        self.df_entrenamiento["education"].replace(educ_order, inplace=True)
        
        
        noyes_mapping_y = {'no': 0, 'yes': 1}
        if 'y' in self.df_entrenamiento.columns:
            self.df_entrenamiento['y'].replace(noyes_mapping_y, inplace=True)
        

        no_yes_unk = {'no': 0, 'yes': 1, 'unknown': 2}
        columns_to_transform = ['default', 'housing', 'loan']
        for column in columns_to_transform:
            self.df_entrenamiento[column].replace(no_yes_unk, inplace=True)
            
        
        var_target = self.df_entrenamiento['y'] if 'y' in self.df_entrenamiento.columns else None
        features_study = self.df_entrenamiento.drop(columns='y', errors='ignore')
        
        features_study = pd.get_dummies(features_study)
        
        norm_numeric = ['age', 'education', 'campaign', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
        scaler = StandardScaler()

        features_study.loc[:, norm_numeric] = scaler.fit_transform(features_study.loc[:, norm_numeric])
        
        if test:
            return features_study, None  
        
        return features_study, var_target  

    
#%%
#Funcion 3 para calcular las componentes principales de los datos contenidos en el dataframe

def pca(data, pvar):

    # Calculamos la matriz de covarianza
    covariance_matrix = np.cov(data, rowvar=False)
    
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
  
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    explained_variance_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    
    n_components = np.argmax(explained_variance_ratio >= pvar) + 1
    
    principal_components = eigenvectors[:, :n_components]
    
    projected_data = np.dot(data, principal_components)
    
 
    columns = [f"Componente_{i+1}" for i in range(n_components)]
    principal_df = pd.DataFrame(data=projected_data, columns=columns)
    
    return principal_df


#%%
#Funcion 4 para calcular la importancia por permutación de cada variable en el dataframe


def feature_importance(x, t, clf, n=None):
    np.random.seed(42)  
    
    baseline_accuracy = accuracy_score(t, clf.predict(x))
    
    importance = []
    for col in x.columns:
        permuted_col = x[col].copy()
        x[col] = np.random.permutation(x[col])  
        
      
        permuted_accuracy = accuracy_score(t, clf.predict(x))
        importance.append(baseline_accuracy - permuted_accuracy)
        
        
        x[col] = permuted_col
    
    importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    if n is not None:
        importance_df = importance_df.head(n)
    
    return importance_df


#%%
#Funcion 5 que entrena y evalúa un clasificador utilizando una única partición train-test de los datos

def train_classifier_single_test(x, t, clf, ptest= 0.3, seed=None):
    
    x_entre, x_ev, y_entre, y_ev = train_test_split(x, t, test_size=ptest, random_state=seed)
    
    
    clf.fit(x_entre, y_entre)
    
    
    strain = clf.score(x_entre, y_entre)
    stest = clf.score(x_ev, y_ev)
    
    return strain, stest, clf


#%%
#Funcion 6 que entrena y evalúa un clasificador utilizando validación cruzada.


def train_classifier_nfold_val(x, t, clf, nfolds=5, seed=None):
    
    s_kfold = StratifiedKFold(n_splits=nfolds, random_state=seed,shuffle=True)
    results = []
    
    # Entrenamiento y evaluación usando validación cruzada
    for train_index, val_index in  s_kfold.split(x, t):
        
        x_entrena, x_val = x.iloc[train_index], x.iloc[val_index]
        t_entrena, t_val = t.iloc[train_index], t.iloc[val_index]
        
        clf_copy = clf.__class__(**clf.get_params())
        clf_copy.fit(x_entrena, t_entrena)
        
        strain = clf_copy.score( x_entrena, t_entrena)
        sval = clf_copy.score(x_val, t_val)
        
        results.append((strain, sval, clf_copy))
    
    return results


#%%
# Funcion 7 que entrena y evalúa un conjunto de clasificadores utilizando validación cruzada

from sklearn.ensemble import BaggingClassifier

def fit_hyperparams(x, t, models, nfolds=5, seed=None):
   
    best_model = None
    best_mean_score = -1  

    res = []

    s_kfold = StratifiedKFold(n_splits=nfolds, random_state=seed, shuffle=True)

    for clf in models:
     
        val = train_classifier_nfold_val(x, t, clf, nfolds=nfolds, seed=seed)

        mean_score = np.mean([val_result[1] for val_result in val])

        res.append((clf, mean_score))

        if mean_score > best_mean_score:
            best_mean_score = mean_score
            best_model = clf

    best_model.fit(x, t)

    return (best_model, res)


#Funcion 8 que entrena  devuelve un conjunto de métricas de evaluación para el clasificador clf sobre los datos x, t


def get_metrics(x, t, clf, target_class=1):
    score = clf.score(x, t)
    cmatrix = confusion_matrix(t, clf.predict(x))
    probs = clf.predict_proba(x)
    fpr, tpr, _ = roc_curve(t, probs[:, 1], pos_label=target_class)
    roc_auc = roc_auc_score(t, probs[:, 1])
    precision, recall, _ = precision_recall_curve(t, probs[:, 1])
    ap = average_precision_score(t, probs[:, 1])

    return score, cmatrix, fpr, tpr, roc_auc, precision, recall, ap

#%%
#Funcion 9 que aplica el clasificador clf a los datos de test x y escribe en el fichero fname las probabilidades asignadas a la clase 1 (yes)

def predict_test(clf, x, fname):
   
    probabilities = clf.predict_proba(x)[:, 1] 
    result_df = pd.DataFrame({'Probability_yes': probabilities})
    result_df.to_csv(fname, index=False) 
    
    return result_df  

#%% 
