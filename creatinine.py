import sqlite3
import pandas as pd
conn = sqlite3.connect(r'C:\Users\rosie\Documents\dissertation_start\dissertation_tables.db') 
cursor = conn.cursor()
creatinine = pd.read_sql_query("SELECT * FROM ICH_Creatinine",conn)

print(creatinine.head())
print(creatinine.shape)
creatinine = creatinine[creatinine['itemid'].isin([51081, 50912])]
creatinine['charttime'] = pd.to_datetime(creatinine['charttime'])
creatinine = creatinine.sort_values(['subject_id','hadm_id','charttime'])
print(creatinine['valueuom'].value_counts(dropna=False))

def aki_finding(admission_data):
    admission_data = admission_data.sort_values('charttime')
    times = list(admission_data['charttime'])
    values = list(admission_data['valuenum'])
    baseline = min(values)
    aki = 0
    for i in range (len(values)):
        for j in range (i+1, len(values)):
            hours = (times[j]-times[i]).total_seconds()/3600
            if hours <= (48):
                if values[j]-values[i]>= 0.3:
                    aki = 1
                    return pd.Series({'AKI':aki})
            if hours <= (7*24):
                if values[j]>=1.5 * baseline:
                    aki = 1
                    return pd.Series({'AKI':aki})
    return pd.Series({'AKI':aki})
aki_labels = creatinine.groupby(['subject_id','hadm_id']).apply(aki_finding).reset_index()
print(aki_labels.head())
print(aki_labels['AKI'].value_counts()) 
aki_labels.to_sql('ICH_AKI_label',conn,if_exists = 'replace',index = False)
conn.close()