#stops warnings being printed
import warnings
warnings.filterwarnings("ignore")
#needed for tabpfn
import os
os.environ["TABPFN_TOKEN"] = "KEY"
#allows tabpfn to run on larger dataset over 1000 
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, RandomizedSearchCV
#import models
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from tabpfn import TabPFNClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import tensorflow as tf
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
from tabpfn_extensions.interpretability.shapiq import get_tabpfn_explainer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, average_precision_score, confusion_matrix
conn = sqlite3.connect(r"C:\Users\rosie\Documents\dissertation_start\dissertation_tables.db")
df = pd.read_sql_query("SELECT * FROM final_dataset",conn)
conn.close()
#convert gender to binary values
df['gender'] = df['gender'].map({'M': 1, 'F': 0})
#separate the predictors and aki labels
X = df.drop(columns = ['subject_id', 'hadm_id', 'stay_id', 'AKI', 'los'])
y = df['AKI']
#split on 80/20 for training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,random_state = 42, stratify = y)

#use the imputer stuff to fill the missing gaps with the median
imputer = SimpleImputer(strategy = "median")
X_train = pd.DataFrame(imputer.fit_transform(X_train),columns = X.columns)
X_test = pd.DataFrame(imputer.transform(X_test),columns = X.columns)

#scale data for logistic regression
scaler = StandardScaler ()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#use random undersampling to reduce majority class
randomundersampler = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = randomundersampler.fit_resample(X_train, y_train)
X_train_scaled_rus = scaler.transform(X_train_rus)
#to calculate and print the metrics, both training and testing for overfittig
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:,1]
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:,1]
    print(name)
    print("Training accuracy:", accuracy_score(y_train, y_train_pred))
    print("Testing accuracy:", accuracy_score(y_test, y_test_pred))
    print("Training F1 Score:", f1_score(y_train, y_train_pred))
    print("Testing F1 Score:", f1_score(y_test, y_test_pred))
    print("Training ROC-AUC:", roc_auc_score(y_train, y_train_prob))
    print("Testing ROC-AUC:", roc_auc_score(y_test, y_test_prob))
    print("Training Precision:", precision_score(y_train, y_train_pred))
    print("Testing Precision:", precision_score(y_test, y_test_pred))
    print("Training Recall:", recall_score(y_train, y_train_pred))
    print("Testing Recall:", recall_score(y_test, y_test_pred))
    print("Training PR-AUC:", average_precision_score(y_train, y_train_prob))
    print("Testing PR-AUC:", average_precision_score(y_test, y_test_prob))
    #calculate specificity for training and testing
    tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, y_train_pred).ravel()
    training_specificity = tn_train / (tn_train + fp_train)
    print("Training Specificity:", training_specificity)
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_test_pred).ravel()
    testing_specificity = tn_test / (tn_test + fp_test)
    print("Testing Specificity:", testing_specificity)
    print(" ")
#ann evaulate model 
def evaluate_ann(name, model, X_train, y_train, X_test, y_test):
    y_train_prob = model.predict(X_train).flatten()
    y_test_prob = model.predict(X_test).flatten()
    y_train_pred = (y_train_prob >= 0.5).astype(int)
    y_test_pred = (y_test_prob >= 0.5).astype(int)
    print(name)
    print("Training accuracy:", accuracy_score(y_train, y_train_pred))
    print("Testing accuracy:", accuracy_score(y_test, y_test_pred))
    print("Training F1 Score:", f1_score(y_train, y_train_pred))
    print("Testing F1 Score:", f1_score(y_test, y_test_pred))
    print("Training ROC-AUC:", roc_auc_score(y_train, y_train_prob))
    print("Testing ROC-AUC:", roc_auc_score(y_test, y_test_prob))
    print("Training Precision:", precision_score(y_train, y_train_pred))
    print("Testing Precision:", precision_score(y_test, y_test_pred))
    print("Training Recall:", recall_score(y_train, y_train_pred))
    print("Testing Recall:", recall_score(y_test, y_test_pred))
    print("Training PR-AUC:", average_precision_score(y_train, y_train_prob))
    print("Testing PR-AUC:", average_precision_score(y_test, y_test_prob))
    tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, y_train_pred).ravel()
    training_specificity = tn_train / (tn_train + fp_train)
    print("Training Specificity:", training_specificity)
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_test_pred).ravel()
    testing_specificity = tn_test / (tn_test + fp_test)
    print("Testing Specificity:", testing_specificity)
    print(" ")

#adding shap
#to create SHAP summary plots for tree based models
def shap_summary_plot(model, X_data, model_name):
    #create tree explainer
    explainer = shap.TreeExplainer(model)
    #calculate shap values for data
    shap_values = explainer.shap_values(X_data)
    #models give shap values in different format so makes sure the plot shows the right class
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values_to_plot = shap_values[:, :, 1]
    else:
        shap_values_to_plot = shap_values
    #plot shap summary plot
    shap.summary_plot(shap_values_to_plot, X_data, show=False)
    plt.title(model_name + " SHAP summary plot")
    plt.tight_layout()
    plt.show()

def ann_summary(model, X_train_data, X_test_data, model_name):
    X_train_df = pd.DataFrame(X_train_data, columns = X.columns)
    X_test_df = pd.DataFrame(X_test_data, columns = X.columns)
    background = shap.sample(X_train_df, 500, random_state=42)
    X_sample = shap.sample(X_test_df, 500, random_state=42)
    explainer = shap.KernelExplainer(lambda x: model.predict(x).flatten(), background)
    shap_values = explainer.shap_values(X_sample, nsamples = 100)
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title(model_name + " SHAP summary plot")
    plt.tight_layout()
    plt.show()
#Logistic regression code
logistic = LogisticRegression(max_iter = 1000)
logistic.fit(X_train_scaled, y_train)
evaluate_model("Logistic regression", logistic, X_train_scaled, y_train, X_test_scaled, y_test)

#logistic regression with random undersampling 
logistic_rus = LogisticRegression(max_iter = 1000)
logistic_rus.fit(X_train_scaled_rus, y_train_rus)
evaluate_model("Logistic regression with random undersampling", logistic_rus, X_train_scaled_rus, y_train_rus, X_test_scaled, y_test)

#XGBoost  
xgb = XGBClassifier(random_state = 42, eval_metric = 'logloss')
xgb.fit(X_train, y_train)
evaluate_model("XGBoost", xgb, X_train, y_train, X_test, y_test)

#xgboost with random under sampling
xgb_rus = XGBClassifier(random_state = 42, eval_metric = 'logloss')
xgb_rus.fit(X_train_rus, y_train_rus)
evaluate_model("XGBoost with random undersampling",xgb_rus, X_train_rus, y_train_rus, X_test, y_test)

#catboost
cat = CatBoostClassifier(random_state = 42, verbose = 0)
cat.fit(X_train, y_train)
evaluate_model("CatBoost", cat, X_train, y_train, X_test, y_test)

#catboost with random undersampling
cat_rus = CatBoostClassifier(random_state = 42, verbose = 0)
cat_rus.fit(X_train_rus, y_train_rus)
evaluate_model("CatBoost with random undersampling", cat_rus, X_train_rus, y_train_rus, X_test, y_test)

#lightgbm
lgbm = LGBMClassifier(random_state = 42)
lgbm.fit(X_train, y_train)
evaluate_model("LightGBM", lgbm, X_train, y_train, X_test, y_test)

#lightgbm with random undersampling
lgbm_rus = LGBMClassifier(random_state = 42)
lgbm_rus.fit(X_train_rus, y_train_rus)
evaluate_model("LightGBM with random undersampling", lgbm_rus, X_train_rus, y_train_rus, X_test, y_test)

#random forest 
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
evaluate_model("Random Forest", rf, X_train, y_train, X_test, y_test)

#random forest with random undersampling 
rf_rus = RandomForestClassifier(random_state=42)
rf_rus.fit(X_train_rus, y_train_rus)
evaluate_model("Random Forest with random undersampling", rf_rus, X_train_rus, y_train_rus, X_test, y_test)
#tabpfn code
tabpfn = TabPFNClassifier(random_state=42, ignore_pretraining_limits=True)
tabpfn.fit(X_train,y_train)
evaluate_model("TabPFN", tabpfn, X_train, y_train, X_test, y_test)

#tabpfn with random undersampling
tabpfn_rus = TabPFNClassifier(random_state=42, ignore_pretraining_limits=True)
tabpfn_rus.fit(X_train_rus,y_train_rus)
evaluate_model("TabPFN with random undersampling", tabpfn_rus, X_train_rus, y_train_rus, X_test, y_test)

#ann 
tf.random.set_seed(42)
np.random.seed(42)
ann = Sequential()
ann.add(Dense(64, input_shape=(X_train_scaled.shape[1],), activation='relu'))
ann.add(Dense(32, activation = 'relu'))
ann.add(Dense(1, activation = 'sigmoid'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy')
ann.fit(X_train_scaled, y_train, epochs = 30, batch_size = 32, verbose = 1, validation_split = 0.1, 
        callbacks = [EarlyStopping(patience = 10, restore_best_weights = True, monitor = 'val_loss')])
evaluate_ann("ANN", ann, X_train_scaled, y_train, X_test_scaled, y_test)

#ann with random undersampling 
ann_rus = Sequential()
ann_rus.add(Dense(64, input_shape=(X_train_scaled_rus.shape[1],), activation='relu'))
ann_rus.add(Dense(32, activation='relu'))
ann_rus.add(Dense(1, activation='sigmoid'))
ann_rus.compile(optimizer='adam', loss='binary_crossentropy')
ann_rus.fit(X_train_scaled_rus, y_train_rus, epochs = 30, batch_size = 32, verbose = 1, validation_split = 0.1, 
        callbacks = [EarlyStopping(patience = 10, restore_best_weights = True, monitor = 'val_loss')])
evaluate_ann("ANN with random undersampling", ann_rus, X_train_scaled_rus, y_train_rus, X_test_scaled, y_test)

#hyperparameter tuning using randomised search cv 5
#logistic regression
logistic_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler()), ("rus", RandomUnderSampler(random_state=42)), ("model", LogisticRegression(max_iter=1000))])
#parameters to try
logistic_params = {"model__C": [0.01, 0.1, 1, 10, 100], "model__solver": ["lbfgs", "liblinear"]}
#search for best logistic regression settings
logistic_search = RandomizedSearchCV(estimator=logistic_pipeline, param_distributions=logistic_params, n_iter=10, cv=5, scoring="roc_auc", random_state=42, n_jobs=-1)
logistic_search.fit(X_train, y_train)
print("Best parameters logistic regression:", logistic_search.best_params_)
best_logistic = logistic_search.best_estimator_
evaluate_model("Tuned logistic regression with random undersampling", best_logistic, X_train, y_train, X_test, y_test)

#xgboost tuning
xgb_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),("rus", RandomUnderSampler(random_state=42)), ("model", XGBClassifier(random_state=42, eval_metric="logloss"))])
#parameters to try for xgboost
xgb_params = {"model__n_estimators": [100, 200, 300], "model__max_depth": [4, 6, 8], "model__learning_rate": [0.01, 0.05, 0.1], "model__subsample": [0.8, 1.0], "model__colsample_bytree": [0.8, 1.0], "model__min_child_weight": [1, 3, 5], "model__gamma": [0, 0.1, 0.3]}
#search for the best xgboost settings
xgb_search = RandomizedSearchCV(estimator=xgb_pipeline, param_distributions=xgb_params, n_iter=20, cv=5, scoring="roc_auc", random_state=42, n_jobs=-1)
xgb_search.fit(X_train, y_train)
print("Best XGBoost parameters:", xgb_search.best_params_)
best_xgb = xgb_search.best_estimator_
evaluate_model("Tuned XGBoost model with random undersampling", best_xgb, X_train, y_train, X_test, y_test)

#catboost tuning
cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),("rus", RandomUnderSampler(random_state=42)), ("model", CatBoostClassifier(random_state=42, verbose=0))])
cat_params = {"model__iterations": [100, 200, 300], "model__depth": [4, 6, 8], "model__learning_rate": [0.01, 0.05, 0.1], "model__l2_leaf_reg": [1, 3, 5, 7]}
#search for best catboost settings
cat_search = RandomizedSearchCV(estimator=cat_pipeline, param_distributions=cat_params, n_iter=20, cv=5, scoring="roc_auc", random_state=42, n_jobs=-1)
cat_search.fit(X_train, y_train)
print("Best CatBoost parameters:", cat_search.best_params_)
best_cat = cat_search.best_estimator_
evaluate_model("Tuned CatBoost model with random undersampling", best_cat, X_train, y_train, X_test, y_test)

#lightgbm tuning
lgbm_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),("rus", RandomUnderSampler(random_state=42)), ("model", LGBMClassifier(random_state=42, verbose=-1))])
#keep feature names for lgbm
lgbm_pipeline.set_output(transform="pandas")
#parameters to try 
lgbm_params = {"model__n_estimators": [100, 200, 300], "model__num_leaves": [31, 63, 127],"model__learning_rate": [0.01, 0.05, 0.1], "model__max_depth": [-1, 5, 10]}
#search for best lgbm settings
lgbm_search = RandomizedSearchCV(estimator=lgbm_pipeline, param_distributions=lgbm_params, n_iter=20, cv=5, scoring="roc_auc", random_state=42, n_jobs=-1)
lgbm_search.fit(X_train, y_train)
print("Best LightGBM parameters:", lgbm_search.best_params_)
best_lgbm = lgbm_search.best_estimator_
evaluate_model("Tuned LightGBM with random undersampling", best_lgbm, X_train, y_train, X_test, y_test)

#random forest tuning
rf_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),("rus", RandomUnderSampler(random_state=42)), ("model", RandomForestClassifier(random_state=42))])
#parameters to try
rf_params = {"model__n_estimators": [100, 200, 300], "model__max_depth": [None, 5, 10, 20], "model__max_features": ["sqrt", "log2"], "model__min_samples_split": [2, 5, 10], "model__min_samples_leaf": [1, 2, 4]}
#search for the best settings 
rf_search = RandomizedSearchCV(estimator=rf_pipeline, param_distributions=rf_params, n_iter=20, cv=5, scoring="roc_auc", random_state=42, n_jobs=-1)
rf_search.fit(X_train, y_train)
print("Best random forest parameters:", rf_search.best_params_)
best_rf = rf_search.best_estimator_
evaluate_model("Tuned Random Forest with random undersampling", best_rf, X_train, y_train, X_test, y_test)
#ann tuning
#function that creates the ANN model
def create_ann_model(optimizer='adam', activation='relu'):
    model = Sequential()
    model.add(Dense(64, input_shape=(X_train_scaled_rus.shape[1],), activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    return model
#turn the Keras model into a sklearn estimator
ann_model = KerasClassifier(model=create_ann_model, verbose=0)
#parameters to try for ANN
ann_params = {"model__optimizer": ["adam", "sgd"], "model__activation": ["relu", "tanh"], "epochs": [30, 50], "batch_size": [16, 32]}
#random search for ANN
ann_search = RandomizedSearchCV(estimator=ann_model, param_distributions=ann_params, n_iter=10, cv=5, scoring="roc_auc", random_state=42, n_jobs=1)
ann_search.fit(X_train_scaled_rus, y_train_rus)
print("Best ANN parameters:", ann_search.best_params_)
best_ann = ann_search.best_estimator_
evaluate_ann("Tuned ANN with random undersampling", best_ann, X_train_scaled_rus, y_train_rus, X_test_scaled, y_test)
#tabpfn with random undersampling inside a pipeline but not tuned
tabpfn_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),("rus", RandomUnderSampler(random_state=42)), ("model", TabPFNClassifier(random_state=42, ignore_pretraining_limits=True))])
tabpfn_pipeline.fit(X_train, y_train)
evaluate_model("TabPFN with random undersampling pipeline", tabpfn_pipeline, X_train, y_train, X_test, y_test)

#SHAP interpretation for tree based models
ann_summary(ann_rus, X_train_scaled_rus, X_test_scaled, "ANN with random undersampling")#
ann_summary(best_ann.model_, X_train_scaled_rus, X_test_scaled, "Tuned ANN with random undersampling")
shap_summary_plot(best_cat.named_steps["model"], X_test, "Tuned CatBoost with random undersampling")
shap_summary_plot(best_rf.named_steps["model"], X_test, "Tuned Random Forest with random undersampling")
shap_summary_plot(best_xgb.named_steps["model"],X_test, "Tuned XGBoost with random undersampling")
shap_summary_plot(best_lgbm.named_steps["model"], X_test, "Tuned LightGBM with random undersampling")