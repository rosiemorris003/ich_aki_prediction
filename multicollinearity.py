import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
conn = sqlite3.connect(r"C:\Users\rosie\Documents\dissertation_start\dissertation_tables.db")
df = pd.read_sql_query("SELECT * FROM final_dataset", conn)
conn.close()
#convert gender to numeric
df["gender"] = df["gender"].map({"M": 1, "F": 0})
X = df.drop(columns=["subject_id", "hadm_id", "stay_id", "AKI"])
X_check = X.select_dtypes(include="number").copy()
#correlation matrix
correlation_matrix = X_check.corr()
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, aspect="auto", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title("Feature correlation matrix after feature reduction")
plt.tight_layout()
plt.savefig("correlation_after.png",dpi=300,bbox_inches="tight")
plt.show()

#find correlations of 0.8 or higher
strong = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        correlation = correlation_matrix.iloc[i, j]
        if abs(correlation) >= 0.8:
            strong.append({"Feature 1": correlation_matrix.columns[i], "Feature 2": correlation_matrix.columns[j], "Correlation": correlation})
strong_df = pd.DataFrame(strong)

if not strong_df.empty:
    strong_df["Absolute correlation"] = (strong_df["Correlation"].abs())
    strong_df = strong_df.sort_values("Absolute correlation",ascending=False)
    print("Strong correlations after feature reduction:")
    print(strong_df.to_string(index=False))
    strong_df.to_csv("correlations_after.csv",index=False)
else:
    print("No correlations of 0.8 or higher were found.")

#median imputation for VIF calculation
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X_check),columns=X_check.columns)

# add constant for VIF calculation
X_vif = add_constant(X_imputed)
vif_values = []
for i in range(X_vif.shape[1]):
    vif = variance_inflation_factor(X_vif.values, i)
    vif_values.append(vif)
vif_results = pd.DataFrame({"Feature": X_vif.columns, "VIF": vif_values})
#remove constant from results
vif_results = vif_results[vif_results["Feature"] != "const"]
vif_results = vif_results.sort_values("VIF", ascending=False)
print("\nVIF values after feature reduction:")
print(vif_results.to_string(index=False))
vif_results.to_csv("vif_after.csv",index=False)