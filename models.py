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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, average_precision_score, confusion_matrix, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
conn = sqlite3.connect(r"C:\Users\rosie\Documents\dissertation_start\dissertation_tables.db")
df = pd.read_sql_query("SELECT * FROM final_dataset",conn)
conn.close()
#convert gender to binary values
df['gender'] = df['gender'].map({'M': 1, 'F': 0})
#separate the predictors and aki labels
X = df.drop(columns = ['subject_id', 'hadm_id', 'stay_id', 'AKI'])
y = df['AKI']
#split on 80/20 for training and testing
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X,y, test_size = 0.2,random_state = 42, stratify = y)
#use the imputer stuff to fill the missing gaps with the median
imputer = SimpleImputer(strategy = "median")
X_train = pd.DataFrame(imputer.fit_transform(X_train_raw),columns = X.columns)
X_test = pd.DataFrame(imputer.transform(X_test_raw),columns = X.columns)

#scale data for logistic regression
scaler = StandardScaler ()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#use random undersampling to reduce majority class
randomundersampler = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = randomundersampler.fit_resample(X_train, y_train)
X_train_scaled_rus = scaler.transform(X_train_rus)
results = []
#to calculate and print the metrics, both training and testing for overfittig
def evaluate_model(name, model, X_train, y_train, X_test, y_test,model_name, model_type, threshold = 0.5):
    y_train_prob = model.predict_proba(X_train)[:,1]
    y_test_prob = model.predict_proba(X_test)[:,1]
    y_train_pred = (y_train_prob >= threshold).astype(int)
    y_test_pred = (y_test_prob >= threshold).astype(int)
    print(name)
    print("Threshold: ", threshold)
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
    print("Training Brier Score:", brier_score_loss(y_train, y_train_prob))
    print("Testing Brier Score:", brier_score_loss(y_test, y_test_prob))
    print(" ")
    results.append({
        "Model": model_name,
        "Type": model_type,
        "Accuracy": accuracy_score(y_test, y_test_pred),
        "ROC-AUC": roc_auc_score(y_test, y_test_prob),
        "PR-AUC": average_precision_score(y_test, y_test_prob),
        "F1 Score": f1_score(y_test, y_test_pred),
        "Precision": precision_score(y_test, y_test_pred),
        "Recall": recall_score(y_test, y_test_pred),
        "Specificity": testing_specificity,
        "Brier Score": brier_score_loss(y_test, y_test_prob)})
#ann evaulate model 
def evaluate_ann(name, model, X_train, y_train, X_test, y_test, model_name, model_type):
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
    print("Training Brier Score:", brier_score_loss(y_train, y_train_prob))
    print("Testing Brier Score:", brier_score_loss(y_test, y_test_prob))
    print(" ")
    results.append({
        "Model": model_name,
        "Type": model_type,
        "Accuracy": accuracy_score(y_test, y_test_pred),
        "ROC-AUC": roc_auc_score(y_test, y_test_prob),
        "PR-AUC": average_precision_score(y_test, y_test_prob),
        "F1 Score": f1_score(y_test, y_test_pred),
        "Precision": precision_score(y_test, y_test_pred),
        "Recall": recall_score(y_test, y_test_pred),
        "Specificity": testing_specificity,
        "Brier Score": brier_score_loss(y_test, y_test_prob)})

#calibrating the models 
def calibrate_and_evaluate_model(name, model, X_train, y_train, X_test, y_test):
    #using sigmoid calibration 
    calibrated_model = CalibratedClassifierCV(estimator=model,method="sigmoid", cv=5)
    calibrated_model.fit(X_train, y_train)
    #use lower threshold for calibration
    evaluate_model(name + " calibrated", calibrated_model, X_train, y_train, X_test, y_test, name, "Calibrated", threshold = 0.5)
    y_test_prob = calibrated_model.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_test_prob, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label=name)
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed proportion of AKI")
    plt.title(name + " calibration curve")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return calibrated_model
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
#odds ratio code for logistic regression 
def odds_ratios(model, feature_names, model_name):
    coefficients = model.coef_[0]
    ratios = np.exp(coefficients)
    odds_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients, "Odds ratio": ratios})
    odds_df["Absolute coefficient"] = odds_df["Coefficient"].abs()
    odds_df = odds_df.sort_values("Absolute coefficient", ascending=False)
    print(model_name + " odds ratios")
    print(odds_df)
    return odds_df
#Logistic regression code
logistic = LogisticRegression(max_iter = 1000)
logistic.fit(X_train_scaled, y_train)
evaluate_model("Logistic regression", logistic, X_train_scaled, y_train, X_test_scaled, y_test, "LR", "Standard")
logistic_odds = odds_ratios(logistic, X.columns, "Logistic regression")

#logistic regression with random undersampling 
logistic_rus = LogisticRegression(max_iter = 1000)
logistic_rus.fit(X_train_scaled_rus, y_train_rus)
evaluate_model("Logistic regression with random undersampling", logistic_rus, X_train_scaled_rus, y_train_rus, X_test_scaled, y_test, "LR", "RUS")
logistic_rus_odds = odds_ratios(logistic_rus, X.columns, "Logistic regression with random undersampling")
#XGBoost  
xgb = XGBClassifier(random_state = 42, eval_metric = 'logloss')
xgb.fit(X_train, y_train)
evaluate_model("XGBoost", xgb, X_train, y_train, X_test, y_test, "XGBoost","Standard")

#xgboost with random under sampling
xgb_rus = XGBClassifier(random_state = 42, eval_metric = 'logloss')
xgb_rus.fit(X_train_rus, y_train_rus)
evaluate_model("XGBoost with random undersampling",xgb_rus, X_train_rus, y_train_rus, X_test, y_test, "XGBoost", "RUS")

#catboost
cat = CatBoostClassifier(random_state = 42, verbose = 0)
cat.fit(X_train, y_train)
evaluate_model("CatBoost", cat, X_train, y_train, X_test, y_test, "Catboost", "Standard")

#catboost with random undersampling
cat_rus = CatBoostClassifier(random_state = 42, verbose = 0)
cat_rus.fit(X_train_rus, y_train_rus)
evaluate_model("CatBoost with random undersampling", cat_rus, X_train_rus, y_train_rus, X_test, y_test, "Catboost", "RUS")

#lightgbm
lgbm = LGBMClassifier(random_state = 42)
lgbm.fit(X_train, y_train)
evaluate_model("LightGBM", lgbm, X_train, y_train, X_test, y_test, "LightGBM", "Standard")

#lightgbm with random undersampling
lgbm_rus = LGBMClassifier(random_state = 42)
lgbm_rus.fit(X_train_rus, y_train_rus)
evaluate_model("LightGBM with random undersampling", lgbm_rus, X_train_rus, y_train_rus, X_test, y_test, "LightGBM", "RUS")

#random forest 
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
evaluate_model("Random Forest", rf, X_train, y_train, X_test, y_test,"Random Forest", "Standard")

#random forest with random undersampling 
rf_rus = RandomForestClassifier(random_state=42)
rf_rus.fit(X_train_rus, y_train_rus)
evaluate_model("Random Forest with random undersampling", rf_rus, X_train_rus, y_train_rus, X_test, y_test, "Random Forest", "RUS")
#tabpfn code
tabpfn = TabPFNClassifier(random_state=42, ignore_pretraining_limits=True)
tabpfn.fit(X_train,y_train)
evaluate_model("TabPFN", tabpfn, X_train, y_train, X_test, y_test, "TabPFN", "Standard")

#tabpfn with random undersampling
tabpfn_rus = TabPFNClassifier(random_state=42, ignore_pretraining_limits=True)
tabpfn_rus.fit(X_train_rus,y_train_rus)
#evaluate_model("TabPFN with random undersampling", tabpfn_rus, X_train_rus, y_train_rus, X_test, y_test, "TabPFN", "RUS")

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
evaluate_ann("ANN", ann, X_train_scaled, y_train, X_test_scaled, y_test, "ANN", "Standard")

#ann with random undersampling 
ann_rus = Sequential()
ann_rus.add(Dense(64, input_shape=(X_train_scaled_rus.shape[1],), activation='relu'))
ann_rus.add(Dense(32, activation='relu'))
ann_rus.add(Dense(1, activation='sigmoid'))
ann_rus.compile(optimizer='adam', loss='binary_crossentropy')
ann_rus.fit(X_train_scaled_rus, y_train_rus, epochs = 30, batch_size = 32, verbose = 1, validation_split = 0.1, 
        callbacks = [EarlyStopping(patience = 10, restore_best_weights = True, monitor = 'val_loss')])
evaluate_ann("ANN with random undersampling", ann_rus, X_train_scaled_rus, y_train_rus, X_test_scaled, y_test, "ANN", "RUS")

#hyperparameter tuning using randomised search cv 5
#logistic regression
logistic_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler()), ("rus", RandomUnderSampler(random_state=42)), ("model", LogisticRegression(max_iter=1000))])
#parameters to try
logistic_params = {"model__C": [0.001, 0.01, 0.1, 1, 10, 100], "model__solver": ["lbfgs", "liblinear"]}
#search for best logistic regression settings
logistic_search = RandomizedSearchCV(estimator=logistic_pipeline, param_distributions=logistic_params, n_iter=12, cv=5, scoring="roc_auc", random_state=42, n_jobs=-1)
logistic_search.fit(X_train_raw, y_train)
print("Best parameters logistic regression:", logistic_search.best_params_)
best_logistic = logistic_search.best_estimator_
evaluate_model("Tuned logistic regression with random undersampling", best_logistic, X_train_raw, y_train, X_test_raw, y_test, "Logistic Regression", "Tuned")
tuned_logistic_model = best_logistic.named_steps["model"]
tuned_logistic_odds = odds_ratios(tuned_logistic_model, X.columns, "Tuned logistic regressiom with random undersampling ")

#xgboost tuning
xgb_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),("rus", RandomUnderSampler(random_state=42)), ("model", XGBClassifier(random_state=42, eval_metric="logloss"))])
#parameters to try for xgboost
xgb_params = {"model__n_estimators": [100, 200, 300, 500], "model__max_depth": [3,4, 6, 8], "model__learning_rate": [0.01, 0.05, 0.1, 0.3], "model__subsample": [0.7, 0.8, 1.0], "model__colsample_bytree": [0.7, 0.8, 1.0], "model__min_child_weight": [1, 3, 5], "model__gamma": [0, 0.1, 0.3]}
#search for the best xgboost settings
xgb_search = RandomizedSearchCV(estimator=xgb_pipeline, param_distributions=xgb_params, n_iter=30, cv=5, scoring="roc_auc", random_state=42, n_jobs=-1)
xgb_search.fit(X_train_raw, y_train)
print("Best XGBoost parameters:", xgb_search.best_params_)
best_xgb = xgb_search.best_estimator_
evaluate_model("Tuned XGBoost model with random undersampling", best_xgb, X_train_raw, y_train, X_test_raw, y_test, "XGBoost", "Tuned")

#catboost tuning
cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),("rus", RandomUnderSampler(random_state=42)), ("model", CatBoostClassifier(random_state=42, verbose=0))])
cat_params = {"model__iterations": [100, 300, 500, 1000], "model__depth": [4, 6, 8, 10], "model__learning_rate": [0.01, 0.03, 0.05, 0.1], "model__l2_leaf_reg": [1, 3, 5, 7, 10]}
#search for best catboost settings
cat_search = RandomizedSearchCV(estimator=cat_pipeline, param_distributions=cat_params, n_iter=25, cv=5, scoring="roc_auc", random_state=42, n_jobs=-1)
cat_search.fit(X_train_raw, y_train)
print("Best CatBoost parameters:", cat_search.best_params_)
best_cat = cat_search.best_estimator_
evaluate_model("Tuned CatBoost model with random undersampling", best_cat, X_train_raw, y_train, X_test_raw, y_test, "Catboost", "Tuned")

#lightgbm tuning
lgbm_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),("rus", RandomUnderSampler(random_state=42)), ("model", LGBMClassifier(random_state=42, verbose=-1))])
#keep feature names for lgbm
lgbm_pipeline.set_output(transform="pandas")
#parameters to try 
lgbm_params = {"model__n_estimators": [100, 200, 300, 500],"model__learning_rate": [0.01, 0.05, 0.1],"model__num_leaves": [15, 31, 63], "model__max_depth": [-1, 4, 6, 8], "model__min_child_samples": [10, 20, 30],"model__subsample": [0.8, 1.0],"model__colsample_bytree": [0.8, 1.0], "model__reg_alpha": [0, 0.1, 1.0], "model__reg_lambda": [0, 0.1, 1.0]}
#search for best lgbm settings
lgbm_search = RandomizedSearchCV(estimator=lgbm_pipeline, param_distributions=lgbm_params, n_iter=30, cv=5, scoring="roc_auc", random_state=42, n_jobs=-1)
lgbm_search.fit(X_train_raw, y_train)
print("Best LightGBM parameters:", lgbm_search.best_params_)
best_lgbm = lgbm_search.best_estimator_
evaluate_model("Tuned LightGBM with random undersampling", best_lgbm, X_train_raw, y_train, X_test_raw, y_test, "LightGBM", "Tuned")

#random forest tuning
rf_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),("rus", RandomUnderSampler(random_state=42)), ("model", RandomForestClassifier(random_state=42))])
#parameters to try
rf_params = {"model__n_estimators": [100, 200, 300, 500], "model__max_depth": [None, 4, 6, 8, 12], "model__max_features": ["sqrt", "log2", None], "model__min_samples_split": [2, 5, 10, 20], "model__min_samples_leaf": [1, 2, 4, 8]}
#search for the best settings 
rf_search = RandomizedSearchCV(estimator=rf_pipeline, param_distributions=rf_params, n_iter=30, cv=5, scoring="roc_auc", random_state=42, n_jobs=-1)
rf_search.fit(X_train_raw, y_train)
print("Best random forest parameters:", rf_search.best_params_)
best_rf = rf_search.best_estimator_
evaluate_model("Tuned Random Forest with random undersampling", best_rf, X_train_raw, y_train, X_test_raw, y_test,"Random Forest", "Tuned")
#ann tuning
#create ANN
def create_ann_model(optimizer="adam", activation="relu"):
    model = Sequential()
    model.add(Dense(64, input_shape=(X.shape[1],),activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=optimizer, loss="binary_crossentropy")
    return model
#put imputation scaling and undersampling in pipeline
early_stopping = EarlyStopping(monitor="val_loss",patience=10,restore_best_weights=True)
ann_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("rus", RandomUnderSampler(random_state=42)),("model", KerasClassifier(model=create_ann_model, verbose=0, random_state=42, validation_split=0.1,callbacks = [early_stopping]))])
#parameters to test
ann_params = {"model__model__optimizer": ["adam", "sgd"], "model__model__activation": ["relu", "tanh"], "model__epochs": [50, 100], "model__batch_size": [16, 32]}
#random search for ANN
ann_search = RandomizedSearchCV(estimator=ann_pipeline, param_distributions=ann_params, n_iter=16, cv=5, scoring="roc_auc", random_state=42, n_jobs=1)
ann_search.fit(X_train_raw, y_train)
print("Best ANN parameters:", ann_search.best_params_)
best_ann = ann_search.best_estimator_
evaluate_model("Tuned ANN with random undersampling", best_ann, X_train_raw, y_train, X_test_raw, y_test, "ANN", "Tuned")
#tabpfn with random undersampling inside a pipeline but not tuned
tabpfn_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),("rus", RandomUnderSampler(random_state=42)), ("model", TabPFNClassifier(random_state=42, ignore_pretraining_limits=True))])
tabpfn_pipeline.fit(X_train_raw, y_train)
evaluate_model("TabPFN with random undersampling pipeline", tabpfn_pipeline, X_train_raw, y_train, X_test_raw, y_test, "TabPFN", "RUS")

#add the calibration to the tuned models 
calibrated_logistic = calibrate_and_evaluate_model("Tuned logistic regression with random undersampling", best_logistic, X_train_raw,y_train, X_test_raw, y_test)
calibrated_xgb = calibrate_and_evaluate_model("Tuned XGBoost with random undersampling", best_xgb, X_train_raw, y_train, X_test_raw, y_test)
calibrated_cat = calibrate_and_evaluate_model("Tuned CatBoost with random undersampling", best_cat, X_train_raw, y_train, X_test_raw, y_test)
calibrated_lgbm = calibrate_and_evaluate_model("Tuned LightGBM with random undersampling", best_lgbm, X_train_raw, y_train, X_test_raw, y_test)
calibrated_rf = calibrate_and_evaluate_model("Tuned Random Forest with random undersampling", best_rf, X_train_raw, y_train, X_test_raw, y_test)
calibrated_tabpfn = calibrate_and_evaluate_model("TabPFN with random undersampling", tabpfn_pipeline, X_train_raw, y_train, X_test_raw, y_test)
calibrated_ann = calibrate_and_evaluate_model("Tuned ANN with random undersampling", best_ann, X_train_raw, y_train, X_test_raw, y_test)

#SHAP for tree models
X_test_cat_shap = pd.DataFrame(best_cat.named_steps["imputer"].transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)
X_test_rf_shap = pd.DataFrame(best_rf.named_steps["imputer"].transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)
X_test_xgb_shap = pd.DataFrame(best_xgb.named_steps["imputer"].transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)
shap_summary_plot(best_cat.named_steps["model"], X_test_cat_shap, "Tuned CatBoost with random undersampling")
shap_summary_plot(best_rf.named_steps["model"], X_test_rf_shap, "Tuned Random Forest with random undersampling")
shap_summary_plot(best_xgb.named_steps["model"],X_test_xgb_shap, "Tuned XGBoost with random undersampling")
#results graph
results_df = pd.DataFrame(results)
results_df.to_csv("model_results.csv", index = False)
graph_results = results_df[results_df["Type"].isin(["Standard", "RUS", "Tuned"])].copy()
graph_results["Graph name"] = (graph_results["Model"]+" - "+graph_results["Type"])
type_colours = {"Standard": "#0072B2","RUS": "#CD853F","Tuned": "#009E73"}
#ROC AUC graph
roc_data = graph_results.sort_values("ROC-AUC", ascending=True)
colours = roc_data["Type"].map(type_colours)
fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(roc_data["Graph name"],roc_data["ROC-AUC"],color=colours)
ax.set_xlabel("ROC-AUC")
ax.set_title("ROC-AUC Across Models")
ax.set_xlim(0, 0.75)
ax.bar_label(bars, fmt="%.3f", padding=3)
plt.tight_layout()
plt.show()

#recall graph
recall_data = graph_results.sort_values("Recall", ascending=True)
colours = recall_data["Type"].map(type_colours)
fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(recall_data["Graph name"],recall_data["Recall"], color=colours)
ax.set_xlabel("Recall")
ax.set_title("Recall Across Models")
ax.set_xlim(0, 0.70)
ax.bar_label(bars, fmt="%.3f", padding=3)
plt.tight_layout()
plt.show()