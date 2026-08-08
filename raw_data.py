import sqlite3
import pandas as pd
#connect to sqlite data base
conn = sqlite3.connect(r'C:\Users\rosie\Documents\dissertation_start\dissertation_tables.db')
#load the smaller mimic-iv files
icd_codes = pd.read_csv(r'C:\Users\rosie\Documents\dissertation_start\d_icd_diagnoses.csv')
diagnoses = pd.read_csv(r'C:\Users\rosie\Documents\dissertation_start\diagnoses_icd.csv')
lab_items = pd.read_csv(r'C:\Users\rosie\Documents\dissertation_start\d_labitems.csv')
chart_items = pd.read_csv(r'C:\Users\rosie\Documents\dissertation_start\d_items.csv')
#load the icu stay information and only keep the columns needed for the project
icu_stays = pd.read_csv(r'C:\Users\rosie\Documents\dissertation_start\icustays.csv.gz',compression ='gzip' )
icu_stays = icu_stays[['subject_id','hadm_id','stay_id','intime','los']]
#convert icu admission time into date time format
icu_stays['intime'] = pd.to_datetime(icu_stays['intime'])
#save icu stay information into sqlite
icu_stays.to_sql('icu_stays',conn,if_exists='replace',index=False)
#item ids needed from the lab and chart event files
lab_items_ids = [51081, 50912, 51265, 51222, 51221, 50971, 51301, 50983, 51279]
chart_item_ids = [220045, 223761, 223762, 224640, 220179, 220180, 220739, 223900, 223901]
#load the weight from inputevents
columns_input = ['subject_id', 'hadm_id', 'stay_id', 'starttime', 'endtime', 'patientweight']
input_events = pd.read_csv(r'C:\Users\rosie\Documents\dissertation_start\inputevents.csv.gz', compression = 'gzip', usecols = columns_input, chunksize = 100000,low_memory=False)
for chunk in input_events:
    #make sure the main id and weight columns are numeric
    chunk['subject_id'] = pd.to_numeric(chunk['subject_id'], errors='coerce')
    chunk['hadm_id'] = pd.to_numeric(chunk['hadm_id'], errors='coerce')
    chunk['stay_id'] = pd.to_numeric(chunk['stay_id'], errors='coerce')
    chunk['patientweight'] = pd.to_numeric(chunk['patientweight'], errors='coerce')
    #add each chunk to the sqlite table
    chunk.to_sql('InputEvents', conn, if_exists='append', index=False)
#load patient age and gender
patients = pd.read_csv(r'patients.csv.gz',compression = 'gzip')
patients = patients[['subject_id','anchor_age','gender']]
patients.to_sql('patients',conn, if_exists='replace',index=False)

#load the chart events needed for the predictor variables
columns_chart = ['subject_id','hadm_id','stay_id','itemid','charttime','valuenum','valueuom']
chart_events = pd.read_csv(r'C:\Users\rosie\Documents\dissertation_start\chartevents.csv.gz',compression='gzip',usecols = columns_chart,chunksize = 100000,low_memory=False)
for chunk in chart_events:
    #only keep clinical measurements needed
    chunk = chunk[chunk['itemid'].isin(chart_item_ids)]
    chunk['subject_id'] = pd.to_numeric(chunk['subject_id'],errors = 'coerce') 
    chunk['hadm_id'] = pd.to_numeric(chunk['hadm_id'],errors = 'coerce')
    chunk['stay_id'] = pd.to_numeric(chunk['stay_id'],errors = 'coerce')
    chunk['itemid'] = pd.to_numeric(chunk['itemid'],errors = 'coerce')
    chunk['valuenum'] = pd.to_numeric(chunk['valuenum'],errors = 'coerce')
    #add filtered chunk to sqlite
    chunk.to_sql('ChartEvents',conn,if_exists = 'append',index=False)

#columns needed from the lab events file
columns_lab= ['labevent_id','subject_id','hadm_id','specimen_id','itemid','charttime','value','valuenum','valueuom','ref_range_lower','ref_range_upper','flag','comments']
lab_events = pd.read_csv(r'labevents.csv.gz',compression = 'gzip', usecols = columns_lab, chunksize=100000,low_memory = False)
for chunk in lab_events:
    #only keep the measurements needed
    chunk = chunk[chunk['itemid'].isin(lab_items_ids)]
    chunk['labevent_id'] = pd.to_numeric(chunk['labevent_id'],errors='coerce')
    chunk['subject_id'] = pd.to_numeric(chunk['subject_id'],errors='coerce')
    chunk['hadm_id'] = pd.to_numeric(chunk['hadm_id'],errors='coerce')
    chunk['specimen_id'] = pd.to_numeric(chunk['specimen_id'],errors='coerce')
    chunk['itemid'] = pd.to_numeric(chunk['itemid'],errors='coerce')
    chunk['valuenum'] = pd.to_numeric(chunk['valuenum'],errors='coerce')
    chunk['ref_range_lower'] = pd.to_numeric(chunk['ref_range_lower'],errors='coerce')
    chunk['ref_range_upper'] = pd.to_numeric(chunk['ref_range_upper'],errors='coerce')
    #add filtered chunk to sqlite
    chunk.to_sql("LabEvents",conn,if_exists = 'append',index=False)
#keep only the columns needed from the look up tables
chart_items = chart_items[['itemid', 'label', 'abbreviation', 'linksto', 'category', 'unitname','param_type', 'lownormalvalue', 'highnormalvalue']]
icd_codes = icd_codes[['icd_code','icd_version','long_title']]
diagnoses = diagnoses[['subject_id', 'hadm_id', 'seq_num', 'icd_code', 'icd_version']]
lab_items = lab_items[['itemid', 'label', 'fluid', 'category']]
#remove any duplicate records before saving the tables
icd_codes=icd_codes.drop_duplicates(subset=['icd_code','icd_version'])
diagnoses = diagnoses.drop_duplicates(subset=['subject_id', 'hadm_id', 'seq_num'])
lab_items = lab_items.drop_duplicates(['itemid'])
#save remaining mimic-iv tables into sqlite
icd_codes.to_sql('ICD_CODES',conn , if_exists = "replace",index=False)
diagnoses.to_sql('DIAGNOSES',conn, if_exists="replace",index=False )
lab_items.to_sql('LabItems',conn, if_exists = "replace",index=False)
chart_items.to_sql('ChartItems',conn,if_exists = "replace",index=False)
#save changes and close database connection
conn.commit()
conn.close()


