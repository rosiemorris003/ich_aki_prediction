import sqlite3
import pandas as pd
sqliteconnection = sqlite3.connect(r'C:\Users\rosie\Documents\dissertation_start\ich_star.db') 
cursor = sqliteconnection.cursor()

query1 = pd.read_sql_query("SELECT DIAGNOSES.* FROM DIAGNOSES JOIN ICD_CODES ON DIAGNOSES.icd_code = ICD_CODES.icd_code AND DIAGNOSES.icd_version = ICD_CODES.icd_version WHERE (ICD_CODES.long_title like 'Nontraumatic intracerebral hemorrhage%' or ICD_CODES.long_title like 'Intracerebral hemorrhage%')", con=sqliteconnection)

print(query1)
print(query1['icd_code'].unique())
#checking creatine
query2 = pd.read_sql_query("SELECT itemid, label FROM LabItems WHERE label LIKE 'creatinine%'",con=sqliteconnection)

query3 = pd.read_sql_query("SELECT itemid, label FROM LabItems WHERE label LIKE 'platelet count%'",con=sqliteconnection)

query4 = pd.read_sql_query("SELECT itemid, label FROM LabItems WHERE label LIKE 'hemoglobin%'",con=sqliteconnection)

query5 = pd.read_sql_query("SELECT itemid, label FROM LabItems WHERE label LIKE 'hematocrit%'",con=sqliteconnection)

query6 = pd.read_sql_query("SELECT itemid, label FROM LabItems WHERE label LIKE 'white blood%'",con=sqliteconnection)

query7 = pd.read_sql_query("SELECT itemid, label FROM LabItems WHERE label LIKE 'potassium%'",con=sqliteconnection)
query8 = pd.read_sql_query("SELECT itemid, label FROM LabItems WHERE label LIKE 'sodium%'",con=sqliteconnection)
query9 = pd.read_sql_query("SELECT itemid,label,category FROM ChartItems WHERE label LIKE 'temperature%'",con=sqliteconnection)
query10 = pd.read_sql_query("SELECT itemid,label FROM ChartItems WHERE label LIKE '%weight%'",con=sqliteconnection)
query11 = pd.read_sql_query("SELECT itemid,label FROM ChartItems WHERE label LIKE '%blood pressure%'",con=sqliteconnection)
query12 = pd.read_sql_query("SELECT itemid,label FROM ChartItems WHERE label LIKE 'heart rate%'",con=sqliteconnection)
query13 = pd.read_sql_query("SELECT icd_code,icd_version,long_title FROM ICD_CODES WHERE long_title LIKE '%hypertension'",con=sqliteconnection)
query14 = pd.read_sql_query("SELECT itemid,label FROM ChartItems WHERE label LIKE 'GCS%'",con=sqliteconnection)
query15 = pd.read_sql_query("SELECT itemid, label FROM LabItems WHERE label LIKE '%red blood%'",con=sqliteconnection)
print(query15)