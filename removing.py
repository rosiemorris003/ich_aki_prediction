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
X = df.drop(columns = ['subject_id', 'hadm_id','stay_id','AKI'])

X_reduced = X.drop(columns=['hypertension'])
y = df['AKI']
#split on 80/20
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42
)

# impute
imputer = SimpleImputer(strategy="median")
X_train_r = pd.DataFrame(imputer.fit_transform(X_train_r), columns=X_reduced.columns)
X_test_r = pd.DataFrame(imputer.transform(X_test_r), columns=X_reduced.columns)

xgb_r = XGBClassifier(random_state = 42, eval_metric = 'logloss')
xgb_r.fit(X_train_r, y_train_r)

y_pred_xgb_r = xgb_r.predict(X_test_r)
y_prob_xgb_r = xgb_r.predict_proba(X_test_r)[:,1]

print("XGBoost (no hypertension)")
print("Accuracy:", accuracy_score(y_test_r, y_pred_xgb_r))
print("F1 Score:", f1_score(y_test_r, y_pred_xgb_r))
print("ROC AUC:", roc_auc_score(y_test_r, y_pred_xgb_r))
print("Precision:", precision_score(y_test_r, y_pred_xgb_r))
print("Recall:", recall_score(y_test_r, y_pred_xgb_r))

