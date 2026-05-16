import os
os.environ["TABPFN_TOKEN"] = "KEY"
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, average_precision_score, confusion_matrix
conn = sqlite3.connect(r"C:\Users\rosie\Documents\dissertation_start\dissertation_tables.db")
df = pd.read_sql_query("SELECT * FROM final_dataset",conn)
conn.close()
#convert gender to binary values
df['gender'] = df['gender'].map({'M': 1, 'F': 0})
#separate the predictors and aki labels
X = df.drop(columns = ['subject_id', 'hadm_id', 'stay_id', 'AKI'])
y = df['AKI']
#split on 80/20 for training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,random_state = 42, stratify = y)

#use the imputer stuff to fill the missing gaps with the median
imputer = SimpleImputer(strategy = "median")
X_train = pd.DataFrame(imputer.fit_transform(X_train),columns = X.columns)
X_test = pd.DataFrame(imputer.transform(X_test),columns = X.columns)

scaler = StandardScaler ()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#use random undersampling to reduce majority class
randomundersampler = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = randomundersampler.fit_resample(X_train, y_train)
X_train_scaled_rus, y_train_scaled_rus = randomundersampler.fit_resample(X_train_scaled, y_train)

#to calculate and print the metrics
def evaluate_model(name, y_test, y_pred, y_prob):
    print(name)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("PR-AUC:", average_precision_score(y_test, y_prob))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    print("Specificity:", specificity)

#Logistic regression code
logistic = LogisticRegression(max_iter = 1000)
logistic.fit(X_train_scaled, y_train)
y_pred_logistic = logistic.predict(X_test_scaled)
y_prob_logistic = logistic.predict_proba(X_test_scaled)[:,1]
evaluate_model("Logistic regression", y_test, y_pred_logistic, y_prob_logistic)

#logistic regression with random undersampling 
logistic_rus = LogisticRegression(max_iter = 1000)
logistic_rus.fit(X_train_scaled_rus, y_train_scaled_rus)
y_pred_logistic_rus = logistic_rus.predict(X_test_scaled)
y_prob_logistic_rus = logistic_rus.predict_proba(X_test_scaled)[:,1]
evaluate_model("Logistic regression with random undersampling", y_test, y_pred_logistic_rus, y_prob_logistic_rus)

#XGBoost  
xgb = XGBClassifier(random_state = 42, eval_metric = 'logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:,1]
evaluate_model("XGBoost", y_test, y_pred_xgb, y_prob_xgb)

#xgboost with random under sampling
xgb_rus = XGBClassifier(random_state = 42, eval_metric = 'logloss')
xgb_rus.fit(X_train_rus, y_train_rus)
y_pred_xgb_rus = xgb_rus.predict(X_test)
y_prob_xgb_rus = xgb_rus.predict_proba(X_test)[:,1]
evaluate_model("XGBoost with random undersampling", y_test, y_pred_xgb_rus, y_prob_xgb_rus)

#catboost
cat = CatBoostClassifier(random_state = 42, verbose = 0)
cat.fit(X_train, y_train)
y_pred_cat = cat.predict(X_test)
y_prob_cat = cat.predict_proba(X_test)[:,1]
evaluate_model("CatBoost", y_test, y_pred_cat, y_prob_cat)

#catboost with random undersampling
cat_rus = CatBoostClassifier(random_state = 42, verbose = 0)
cat_rus.fit(X_train_rus, y_train_rus)
y_pred_cat_rus = cat_rus.predict(X_test)
y_prob_cat_rus = cat_rus.predict_proba(X_test)[:,1]
evaluate_model("CatBoost with random undersampling", y_test, y_pred_cat_rus, y_prob_cat_rus)

#lightgbm
lgbm = LGBMClassifier(random_state = 42)
lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_test)
y_prob_lgbm = lgbm.predict_proba(X_test)[:,1]
evaluate_model("LightGBM", y_test, y_pred_lgbm, y_prob_lgbm)

#lightgbm with random undersampling
lgbm_rus = LGBMClassifier(random_state = 42)
lgbm_rus.fit(X_train_rus, y_train_rus)
y_pred_lgbm_rus = lgbm_rus.predict(X_test)
y_prob_lgbm_rus = lgbm_rus.predict_proba(X_test)[:,1]
evaluate_model("LightGBM with random undersampling", y_test, y_pred_lgbm_rus, y_prob_lgbm_rus)

#tabpfn code
tabpfn = TabPFNClassifier(random_state=42, ignore_pretraining_limits=True)
tabpfn.fit(X_train,y_train)
y_pred_tabpfn = tabpfn.predict(X_test)
y_prob_tabpfn = tabpfn.predict_proba(X_test)[:,1]
evaluate_model("TabPFN", y_test, y_pred_tabpfn, y_prob_tabpfn)

#tabpfn with random undersampling
tabpfn_rus = TabPFNClassifier(random_state=42, ignore_pretraining_limits=True)
tabpfn_rus.fit(X_train_rus,y_train_rus)
y_pred_tabpfn_rus = tabpfn_rus.predict(X_test)
y_prob_tabpfn_rus = tabpfn_rus.predict_proba(X_test)[:,1]
evaluate_model("TabPFN with random undersampling", y_test, y_pred_tabpfn_rus, y_prob_tabpfn_rus)
