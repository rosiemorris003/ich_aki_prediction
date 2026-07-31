import sqlite3
import pandas as pd
conn = sqlite3.connect(r'C:\Users\rosie\Documents\dissertation_start\dissertation_tables.db')
cursor = conn.cursor()
creatinine = pd.read_sql_query("SELECT * FROM ICH_Creatinine",conn)
print(creatinine.head())
print(creatinine.shape)
creatinine = creatinine[creatinine['itemid'].isin([51081, 50912])]
creatinine['charttime'] = pd.to_datetime(creatinine['charttime'])
creatinine['intime'] = pd.to_datetime(creatinine['intime'])
creatinine['hours_from_icu'] = (creatinine['charttime']-creatinine['intime']).dt.total_seconds() / 3600
creatinine = creatinine[creatinine['hours_from_icu']>=-(7 * 24)].copy()
creatinine = creatinine.sort_values(['subject_id', 'hadm_id', 'charttime'])
print(creatinine['valueuom'].value_counts(dropna=False))

def aki_finding(admission_data):
    admission_data = admission_data.sort_values('charttime')
    admission_data = admission_data.dropna(subset=['valuenum'])
    if len(admission_data)<2:
        return pd.Series({'AKI_first_24h':0,'AKI_after_24h':0})
    first_24h = admission_data[admission_data['hours_from_icu']<=24]
    after_24h = admission_data[admission_data['hours_from_icu']>24]
    aki_first_24h = 0
    if len(first_24h)>=2:
        times = list(first_24h['charttime'])
        values = list(first_24h['valuenum'])
        hours_from_icu = list(first_24h['hours_from_icu'])

        for i in range(len(values)):
            for j in range(i+1, len(values)):
                hours = (times[j]-times[i]).total_seconds()/3600
                if hours_from_icu[j]>=0 and hours_from_icu[j]<=24:

                    if hours<=48 and values[j]-values[i]>=0.3:
                        aki_first_24h = 1

                    if hours<=7*24 and values[j]>=1.5*values[i]:
                        aki_first_24h = 1
    aki_after_24h = 0
    if len(after_24h)>=1:
        times = list(admission_data['charttime'])
        values = list(admission_data['valuenum'])
        hours_from_icu = list(admission_data['hours_from_icu'])
        for i in range(len(values)):
            for j in range(i+1, len(values)):
                hours_between = (times[j]-times[i]).total_seconds()/3600
                if hours_from_icu[j]>24:
                    if (hours_between<=48 and values[j]-values[i]>=0.3):
                        aki_after_24h = 1
                    if (hours_between<=7*24 and values[j]>=1.5*values[i]):
                        aki_after_24h = 1

    return pd.Series({'AKI_first_24h': aki_first_24h, 'AKI_after_24h': aki_after_24h})

aki_labels = (creatinine.groupby(['subject_id', 'hadm_id']).apply(aki_finding).reset_index())
print(aki_labels.head())
print(aki_labels[['AKI_first_24h', 'AKI_after_24h']].value_counts())
aki_labels_main = aki_labels[aki_labels['AKI_first_24h'] == 0].copy()
print(aki_labels_main.value_counts())
aki_labels.to_sql('ICH_AKI_label_all', conn, if_exists='replace', index=False)
aki_labels_main = aki_labels_main.rename(columns={'AKI_after_24h': 'AKI'})
aki_labels_main = aki_labels_main[['subject_id', 'hadm_id', 'AKI']]
aki_labels_main.to_sql('ICH_AKI_label',conn,if_exists='replace',index=False)
conn.close()