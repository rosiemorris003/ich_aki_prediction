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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, average_precision_score, confusion_matrix


conn = sqlite3.connect(r"C:\Users\rosie\Documents\dissertation_start\dissertation_tables.db")
df = pd.read_sql_query("SELECT * FROM final_dataset",conn)
conn.close()
df['gender'] = df['gender'].map({'M': 1, 'F': 0})
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])
X = df.drop(columns = ['subject_id', 'hadm_id', 'stay_id', 'AKI'])
y = df['AKI']
#split on 80/20
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,random_state = 42)

#use the imputer stuff to fill the missing gaps with the median
imputer = SimpleImputer(strategy = "median")
X_train = pd.DataFrame(imputer.fit_transform(X_train),columns = X.columns)
X_test = pd.DataFrame(imputer.transform(X_test),columns = X.columns)

scaler = StandardScaler ()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Logistic regression code
logistic = LogisticRegression(max_iter = 1000)
logistic.fit(X_train_scaled, y_train)

y_pred_logistic = logistic.predict(X_test_scaled)
y_prob_logistic = logistic.predict_proba(X_test_scaled)[:,1]
print("Logistic regression")
print("Accuracy:", accuracy_score(y_test, y_pred_logistic ))
print("F1 Score:", f1_score(y_test, y_pred_logistic ))
print("ROC AUC:", roc_auc_score(y_test, y_prob_logistic))
print("Precision:", precision_score(y_test, y_pred_logistic ))
print("Recall:", recall_score(y_test, y_pred_logistic ))
print("PR-AUC:", average_precision_score(y_test, y_prob_logistic))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_logistic ).ravel()
specificity = tn / (tn + fp)
print("Specificity:", specificity)

#XGBoost code now 
xgb = XGBClassifier(random_state = 42, eval_metric = 'logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:,1]

print("XGBoost")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("F1 Score:", f1_score(y_test, y_pred_xgb))
print("ROC AUC:", roc_auc_score(y_test, y_prob_xgb))
print("Precision:", precision_score(y_test, y_pred_xgb))
print("Recall:", recall_score(y_test, y_pred_xgb))
print("PR-AUC:", average_precision_score(y_test,y_prob_xgb))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_xgb).ravel()
specificity = tn/ (tn+fp)
print("Specificity:",specificity)

#catboost
cat = CatBoostClassifier(random_state = 42, verbose = 0)
cat.fit(X_train, y_train)
y_pred_cat = cat.predict(X_test)
y_prob_cat = cat.predict_proba(X_test)[:,1]
print("CatBoost")
print("Accuracy:", accuracy_score(y_test, y_pred_cat))
print("F1 Score:", f1_score(y_test, y_pred_cat))
print("ROC AUC:", roc_auc_score(y_test, y_prob_cat))
print("Precision:", precision_score(y_test, y_pred_cat))
print("Recall:", recall_score(y_test, y_pred_cat))
print("PR-AUC:", average_precision_score(y_test,y_prob_cat))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_cat).ravel()
specificity = tn/ (tn+fp)
print("Specificity:",specificity)

#lightgbm
lgbm = LGBMClassifier(random_state = 42)
lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_test)
y_prob_lgbm = lgbm.predict_proba(X_test)[:,1]
print("LightGBM")
print("Accuracy:", accuracy_score(y_test, y_pred_lgbm))
print("F1 Score:", f1_score(y_test, y_pred_lgbm))
print("ROC AUC:", roc_auc_score(y_test, y_prob_lgbm))
print("Precision:", precision_score(y_test, y_pred_lgbm))
print("Recall:", recall_score(y_test, y_pred_lgbm))
print("PR-AUC:", average_precision_score(y_test, y_prob_lgbm))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_lgbm).ravel()
specificity = tn / (tn + fp)
print("Specificity:", specificity)

#tabpfn code
tabpfn = TabPFNClassifier(random_state=42, ignore_pretraining_limits=True)
tabpfn.fit(X_train,y_train)
y_pred_tabpfn = tabpfn.predict(X_test)
y_prob_tabpfn = tabpfn.predict_proba(X_test)[:,1]
print("TabPFN")
print("Accuracy:", accuracy_score(y_test, y_pred_tabpfn))
print("F1 Score:", f1_score(y_test, y_pred_tabpfn))
print("ROC AUC:", roc_auc_score(y_test, y_prob_tabpfn))
print("Precision:", precision_score(y_test, y_pred_tabpfn))
print("Recall:", recall_score(y_test, y_pred_tabpfn))
print("PR-AUC:", average_precision_score(y_test, y_prob_tabpfn))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_tabpfn).ravel()
specificity = tn / (tn + fp)
print("Specificity:", specificity)

