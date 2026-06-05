import sqlite3
import pandas as pd
conn = sqlite3.connect(r"C:\Users\rosie\Documents\dissertation_start\dissertation_tables.db")
#check the total number of rows in the final dataset 
final = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset", conn)
print("Final dataset rows", final.iloc[0]['COUNT(*)'])

#check for duplicate stays in final dataset
duplicate = pd.read_sql_query("SELECT COUNT(*) FROM (SELECT stay_id FROM Final_Dataset GROUP BY stay_id HAVING COUNT(*)>1)",conn)
print("Duplicates", duplicate.iloc[0]['COUNT(*)'])

#check how many patients develop aki and how many do not 
aki = pd.read_sql_query("SELECT AKI, COUNT(*) FROM Final_Dataset GROUP BY AKI", conn)
print(aki)

#check there are no readings after 24 hours in the lab and chart events
lab_24h = pd.read_sql_query("SELECT COUNT(*) FROM ICH_LabEvents_24h l JOIN ICH_FIRST_ICU_STAY i ON l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id WHERE l.charttime >= datetime(i.intime, '+24hours')", conn)
print("Readings after the 24hours in labs: ",lab_24h.iloc[0]["COUNT(*)"])
chart_24h = pd.read_sql_query("SELECT COUNT(*) FROM ICH_ChartEvents_24h c JOIN ICH_FIRST_ICU_STAY i on c.subject_id = i.subject_id AND c.hadm_id = i.hadm_id WHERE c.charttime >= datetime(i.intime,'+24hours')",conn)
print("Readings after the 24hours in chart events:", chart_24h.iloc[0]['COUNT(*)'])

#check there are no readings before the icu admission in the lab or chart events
lab_before = pd.read_sql_query("SELECT COUNT(*) FROM ICH_LabEvents_24h l JOIN ICH_FIRST_ICU_STAY i on l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id WHERE l.charttime < i.intime",conn)
print("Lab readings before ICU admission: ", lab_before.iloc[0]['COUNT(*)'])
chart_before = pd.read_sql_query("SELECT COUNT(*) FROM ICH_ChartEvents_24h c JOIN ICH_FIRST_ICU_STAY i on c.subject_id = i.subject_id AND c.hadm_id = i.hadm_id WHERE c.charttime < i.intime",conn)
print("Chart readings before icu admission:", chart_before.iloc[0]['COUNT(*)'])

#check for any unrealistic temperatures
temperature = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE temperature_min < 30 OR temperature_max > 45", conn)
print("Unrealistic temperature count: ", temperature.iloc[0]['COUNT(*)'])

#check for any unrealistic heart rates
heart_rate = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE heart_rate_min <20 OR heart_rate_max > 250",conn)
print("Unrealistic heart rate readings: ", heart_rate.iloc[0]['COUNT(*)'])

#check for any unrealistic blood pressure values 
blood_pressure = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE sbp_min <30 OR sbp_max >300 or dbp_min <10 or dbp_max >200",conn)
print("Unrealistic blood pressure readings: ", blood_pressure.iloc[0]['COUNT(*)'])

#check gcs is in between 3-15
gcs = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE gcs_min > 15 or gcs_min < 3",conn)
print("Unrealistic GCS values: ", gcs.iloc[0]['COUNT(*)'])

#check the weight values are realistic
weight = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE weight_mean < 30 or weight_mean >300", conn)
print("Unrealistic weight values: ", weight.iloc[0]['COUNT(*)'])

#check only adults are included 
adults = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE anchor_age <18",conn)
print("Number of patients below 18:", adults.iloc[0]['COUNT(*)'])

#check every patient included has an aki label 
aki_label = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE AKI is NULL", conn)
print("Number of patients who do not have an AKI label: ", aki_label.iloc[0]['COUNT(*)'])
#check missing heart rate values
heart_rate_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE heart_rate_mean IS NULL OR heart_rate_min IS NULL OR heart_rate_max IS NULL", conn)
print("Missing heart rate values:", heart_rate_null.iloc[0]['COUNT(*)'])

#check missing systolic blood pressure values
sbp_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE sbp_mean IS NULL OR sbp_min IS NULL OR sbp_max IS NULL", conn)
print("Missing systolic BP values:", sbp_null.iloc[0]['COUNT(*)'])

#check missing diastolic blood pressure values
dbp_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE dbp_mean IS NULL OR dbp_min IS NULL OR dbp_max IS NULL", conn)
print("Missing diastolic BP values:", dbp_null.iloc[0]['COUNT(*)'])

#check missing temperature values
temperature_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE temperature_mean IS NULL OR temperature_min IS NULL OR temperature_max IS NULL", conn)
print("Missing temperature values:", temperature_null.iloc[0]['COUNT(*)'])

#check missing weight values
weight_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE weight_mean IS NULL", conn)
print("Missing weight values:", weight_null.iloc[0]['COUNT(*)'])

#check missing GCS values
gcs_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE gcs_min IS NULL", conn)
print("Missing GCS values:", gcs_null.iloc[0]['COUNT(*)'])

#check missing platelet values
platelets_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE platelets_mean IS NULL OR platelets_min IS NULL OR platelets_max IS NULL", conn)
print("Missing platelet values:", platelets_null.iloc[0]['COUNT(*)'])

#check missing haemoglobin values
haemoglobin_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE haemoglobin_mean IS NULL OR haemoglobin_min IS NULL OR haemoglobin_max IS NULL", conn)
print("Missing haemoglobin values:", haemoglobin_null.iloc[0]['COUNT(*)'])

#check missing haematocrit values
haematocrit_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE haematocrit_mean IS NULL OR haematocrit_min IS NULL OR haematocrit_max IS NULL", conn)
print("Missing haematocrit values:", haematocrit_null.iloc[0]['COUNT(*)'])

#check missing potassium values
potassium_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE potassium_mean IS NULL OR potassium_min IS NULL OR potassium_max IS NULL", conn)
print("Missing potassium values:", potassium_null.iloc[0]['COUNT(*)'])

#check missing sodium values
sodium_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE sodium_mean IS NULL OR sodium_min IS NULL OR sodium_max IS NULL", conn)
print("Missing sodium values:", sodium_null.iloc[0]['COUNT(*)'])

#check missing white blood cell values
wbc_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE wbc_mean IS NULL OR wbc_min IS NULL OR wbc_max IS NULL", conn)
print("Missing white blood cell values:", wbc_null.iloc[0]['COUNT(*)'])

#check missing red blood cell values
rbc_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE rbc_mean IS NULL OR rbc_min IS NULL OR rbc_max IS NULL", conn)
print("Missing red blood cell values:", rbc_null.iloc[0]['COUNT(*)'])

#check missing hypertension values
hypertension_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE hypertension_count IS NULL", conn)
print("Missing hypertension values:", hypertension_null.iloc[0]['COUNT(*)'])

#check missing length of stay values
los_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE los IS NULL", conn)
print("Missing LOS values:", los_null.iloc[0]['COUNT(*)'])

#check for negative length of stay values
los_negative = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE los<0",conn)
print("Negative LOS values: ", los_negative.iloc[0]['COUNT(*)'])

#check for any duplicate patients
duplicate_patients = pd.read_sql_query("SELECT COUNT(*) FROM (SELECT subject_id FROM Final_Dataset GROUP BY subject_id HAVING COUNT(*) > 1)", conn)
print("Duplicate patients:", duplicate_patients.iloc[0]['COUNT(*)'])

#check only 0 and 1 are used for AKI labels
aki_values = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE AKI != 0 AND AKI !=1", conn)
print("Invalid AKI values:", aki_values.iloc[0]['COUNT(*)'])

#check gender has only male and female
gender_values = pd.read_sql_query("SELECT gender, COUNT(*) FROM Final_Dataset GROUP BY gender", conn)
print("Genders:", gender_values)

#check hypertension is not negative
hypertension_negative = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE hypertension_count < 0", conn)
print("Negative hypertension values:", hypertension_negative.iloc[0]['COUNT(*)'])

#check LOS is not extremely high
los_extreme = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE los > 365", conn)
print("Extreme los values:", los_extreme.iloc[0]['COUNT(*)'])