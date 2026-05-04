import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
conn = sqlite3.connect(r"C:\Users\rosie\Documents\dissertation_start\dissertation_tables.db")
df = pd.read_sql_query("SELECT * FROM final_dataset",conn)

conn.close()

df['AKI'] = df['AKI'].fillna(0)
df['gender'] = df['gender'].map({'M': 1, 'F': 0})
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])
id_cols = ['subject_id', 'hadm_id', 'stay_id', 'AKI']

lab_cols = [col for col in df.columns if any(lab in col for lab in [
    'haemoglobin', 'haematocrit', 'platelet',
    'wbc', 'rbc'
])]

X = df.drop(columns=id_cols + lab_cols)
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

y_pred = logistic.predict(X_test_scaled)
y_prob = logistic.predict_proba(X_test_scaled)[:,1]
#print("Logistic regression")
#print("Accuracy:", accuracy_score(y_test, y_pred))
#print("F1 Score:", f1_score(y_test, y_pred))
#print("ROC AUC:", roc_auc_score(y_test, y_prob))
#print("Precision:", precision_score(y_test, y_pred))
#print("Recall:", recall_score(y_test, y_pred))

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

cat = CatBoostClassifier(random_state = 42, verbose = 0)
cat.fit(X_train, y_train)
y_pred_cat = cat.predict(X_test)
y_prob_cat = cat.predict_proba(X_test)[:,1]
#print("CatBoost")
#print("Accuracy:", accuracy_score(y_test, y_pred_cat))
#print("F1 Score:", f1_score(y_test, y_pred_cat))
#print("ROC AUC:", roc_auc_score(y_test, y_prob_cat))
#print("Precision:", precision_score(y_test, y_pred_cat))
#print("Recall:", recall_score(y_test, y_pred_cat))
lgbm = LGBMClassifier(random_state = 42)
lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_test)
y_prob_lgbm = lgbm.predict_proba(X_test)[:,1]
#print("LightGBM")
#print("Accuracy:", accuracy_score(y_test, y_pred_lgbm))
#print("F1 Score:", f1_score(y_test, y_pred_lgbm))
#print("ROC AUC:", roc_auc_score(y_test, y_prob_lgbm))
#print("Precision:", precision_score(y_test, y_pred_lgbm))
#print("Recall:", recall_score(y_test, y_pred_lgbm))