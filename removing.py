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
conn = sqlite3.connect(r"C:\Users\rosie\Documents\dissertation_start\ich_star.db")
df = pd.read_sql_query("SELECT * FROM final_with_age_gender",conn)

conn.close()

df['AKI'] = df['AKI'].fillna(0)
df['gender'] = df['gender'].map({'M': 1, 'F': 0})
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])
X = df.drop(columns = ['subject_id', 'hadm_id', 'AKI'])

X_reduced = X.drop(columns='gender')
y = df['AKI']
#split on 80/20
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42
)

# impute
imputer = SimpleImputer(strategy="median")
X_train_r = pd.DataFrame(imputer.fit_transform(X_train_r), columns=X_reduced.columns)
X_test_r = pd.DataFrame(imputer.transform(X_test_r), columns=X_reduced.columns)

# scale
scaler = StandardScaler()
X_train_r = scaler.fit_transform(X_train_r)
X_test_r = scaler.transform(X_test_r)

lgbm_r = LGBMClassifier(random_state=42)
lgbm_r.fit(X_train_r, y_train_r)

y_pred_lgbm_r = lgbm_r.predict(X_test_r)
y_prob_lgbm_r = lgbm_r.predict_proba(X_test_r)[:,1]

print("LightGBM (no hypertension)")
print("Accuracy:", accuracy_score(y_test_r, y_pred_lgbm_r))
print("F1 Score:", f1_score(y_test_r, y_pred_lgbm_r))
print("ROC AUC:", roc_auc_score(y_test_r, y_prob_lgbm_r))
print("Precision:", precision_score(y_test_r, y_pred_lgbm_r))
print("Recall:", recall_score(y_test_r, y_pred_lgbm_r))


print([col for col in X.columns if 'potassium' in col.lower()])
print([col for col in X.columns if 'sodium' in col.lower()])
print([col for col in X_reduced.columns if 'potassium' in col.lower()])
print([col for col in X_reduced.columns if 'sodium' in col.lower()])