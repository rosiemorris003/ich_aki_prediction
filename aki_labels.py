import sqlite3
import pandas as pd
#connect to SQLite database and load creatinine data
db_path = 'dissertation_tables.db'
conn = sqlite3.connect(db_path)
creatinine = pd.read_sql_query("SELECT * FROM ICH_Creatinine",conn)
#keep serum creatinine item used for aki definition
creatinine = creatinine[creatinine['itemid'].isin([50912])]
#convert the times so they can be compared with the icu admission time
creatinine['charttime'] = pd.to_datetime(creatinine['charttime'])
creatinine['intime'] = pd.to_datetime(creatinine['intime'])
#workout when each creatinine reading was taken relative to the ICU admission
creatinine['hours_from_icu'] = (creatinine['charttime']-creatinine['intime']).dt.total_seconds() / 3600
#only keep the readings from up to the 7 days before admission and onwards
creatinine = creatinine[creatinine['hours_from_icu']>=-(7 * 24)].copy()
#make sure each of the patients readings are in time order
creatinine = creatinine.sort_values(['subject_id', 'hadm_id', 'charttime'])
#count the number of usable readings for each admission
measurement_counts = (creatinine.dropna(subset=['valuenum']).groupby(['subject_id', 'hadm_id']).agg(total_readings=('valuenum', 'count'), readings_after_24h=('hours_from_icu',lambda x: (x > 24).sum())).reset_index())
#the patient needs at least two readings and one after 24 hours to be eligible
eligible = measurement_counts[(measurement_counts['total_readings'] >= 2) &(measurement_counts['readings_after_24h'] >= 1)][['subject_id', 'hadm_id']]
creatinine = creatinine.merge(eligible,on=['subject_id', 'hadm_id'],how='inner')

#check each admission for aki using the creatinine criteria
def aki_finding(admission_data):
    admission_data = admission_data.sort_values('charttime')
    admission_data = admission_data.dropna(subset=['valuenum'])
    #aki cannot be checked properly with fewer than two readings
    if len(admission_data)<2:
        return pd.Series({'AKI_first_24h':0,'AKI_after_24h':0})
    #separate the readings from before 24 hours and after that point
    first_24h = admission_data[admission_data['hours_from_icu']<=24]
    after_24h = admission_data[admission_data['hours_from_icu']>24]
    #check whether AKI has developed over the first 24 hours
    aki_first_24h = 0
    if len(first_24h)>=2:
        times = list(first_24h['charttime'])
        values = list(first_24h['valuenum'])
        hours_from_icu = list(first_24h['hours_from_icu'])

        #compare each later creatinine reading with earlier readings
        for i in range(len(values)):
            for j in range(i+1, len(values)):
                hours = (times[j]-times[i]).total_seconds()/3600
                if hours_from_icu[j]>=0 and hours_from_icu[j]<=24:
                    #increase of at least 0.3 mg/dL within 48 hours
                    if hours<=48 and values[j]-values[i]>=0.3:
                        aki_first_24h = 1
                    # increase of at least 1.5 times a previous value within 7 days
                    if hours<=7*24 and values[j]>=1.5*values[i]:
                        aki_first_24h = 1
    #now check for if aki developed after first 24 hours
    aki_after_24h = 0
    if len(after_24h)>=1:
        times = list(admission_data['charttime'])
        values = list(admission_data['valuenum'])
        hours_from_icu = list(admission_data['hours_from_icu'])
        for i in range(len(values)):
            for j in range(i+1, len(values)):
                hours_between = (times[j]-times[i]).total_seconds()/3600
                #only label aki if the reading is after 24 hours
                if hours_from_icu[j]>24:
                    if (hours_between<=48 and values[j]-values[i]>=0.3):
                        aki_after_24h = 1
                    if (hours_between<=7*24 and values[j]>=1.5*values[i]):
                        aki_after_24h = 1

    return pd.Series({'AKI_first_24h': aki_first_24h, 'AKI_after_24h': aki_after_24h})

#run aki check separately for each hospital admission
aki_labels = (creatinine.groupby(['subject_id', 'hadm_id']).apply(aki_finding).reset_index())
#check number of patients in each aki group
print(aki_labels[['AKI_first_24h', 'AKI_after_24h']].value_counts())
#remove any patients who have already met the aki criteria in the first 24 hours
aki_labels_main = aki_labels[aki_labels['AKI_first_24h'] == 0].copy()
print(aki_labels_main["AKI_after_24h"].value_counts())
#save all labels so we can check the early aki cases if we need
aki_labels.to_sql('ICH_AKI_label_all', conn, if_exists='replace', index=False)
#use aki after 24 hours as the final outcome
aki_labels_main = aki_labels_main.rename(columns={'AKI_after_24h': 'AKI'})
aki_labels_main = aki_labels_main[['subject_id', 'hadm_id', 'AKI']]
#save final aki labels back to the sqlite table
aki_labels_main.to_sql('ICH_AKI_label',conn,if_exists='replace',index=False)