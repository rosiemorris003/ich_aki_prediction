import sqlite3
import pandas as pd
db_path = 'dissertation_tables.db'
conn = sqlite3.connect(db_path)
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
lab_24h = pd.read_sql_query("SELECT COUNT(*) FROM ICH_LabEvents_24h l JOIN ICH_FIRST_ICU_STAY i ON l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id WHERE l.charttime >= datetime(i.intime, '+24 hours')", conn)
print("Readings after the 24hours in labs: ",lab_24h.iloc[0]["COUNT(*)"])
chart_24h = pd.read_sql_query("SELECT COUNT(*) FROM ICH_ChartEvents_24h c JOIN ICH_FIRST_ICU_STAY i on c.subject_id = i.subject_id AND c.hadm_id = i.hadm_id WHERE c.charttime >= datetime(i.intime,'+24 hours')",conn)
print("Readings after the 24hours in chart events:", chart_24h.iloc[0]['COUNT(*)'])

#check there are no readings before the icu admission in the lab or chart events
lab_before = pd.read_sql_query("SELECT COUNT(*) FROM ICH_LabEvents_24h l JOIN ICH_FIRST_ICU_STAY i on l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id WHERE l.charttime < i.intime",conn)
print("Lab readings before ICU admission: ", lab_before.iloc[0]['COUNT(*)'])
chart_before = pd.read_sql_query("SELECT COUNT(*) FROM ICH_ChartEvents_24h c JOIN ICH_FIRST_ICU_STAY i on c.subject_id = i.subject_id AND c.hadm_id = i.hadm_id WHERE c.charttime < i.intime",conn)
print("Chart readings before icu admission:", chart_before.iloc[0]['COUNT(*)'])

#check for any unrealistic temperatures
temperature = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE temperature_min < 20 OR temperature_mean>50", conn)
print("Unrealistic temperature count: ", temperature.iloc[0]['COUNT(*)'])

#check for any unrealistic heart rates
heart_rate = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE heart_rate_max > 300 OR heart_rate_mean<10",conn)
print("Unrealistic heart rate readings: ", heart_rate.iloc[0]['COUNT(*)'])

#check for any unrealistic blood pressure values 
blood_pressure = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE sbp_min <20 OR sbp_max >350 or dbp_min <5 or dbp_max >250",conn)
print("Unrealistic blood pressure readings: ", blood_pressure.iloc[0]['COUNT(*)'])

#check gcs is in between 3-15
gcs = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE gcs_min > 15 or gcs_min < 3",conn)
print("Unrealistic GCS values: ", gcs.iloc[0]['COUNT(*)'])

#check the weight values are realistic
weight = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE weight_mean < 20 or weight_mean >400", conn)
print("Unrealistic weight values: ", weight.iloc[0]['COUNT(*)'])

#check only adults are included 
adults = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE anchor_age <18",conn)
print("Number of patients below 18:", adults.iloc[0]['COUNT(*)'])

#check every patient included has an aki label 
aki_label = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE AKI is NULL", conn)
print("Number of patients who do not have an AKI label: ", aki_label.iloc[0]['COUNT(*)'])
#check missing heart rate values
heart_rate_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE heart_rate_mean IS NULL OR heart_rate_max IS NULL", conn)
print("Missing heart rate values:", heart_rate_null.iloc[0]['COUNT(*)'])

#check missing systolic blood pressure values
sbp_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE sbp_mean IS NULL OR sbp_min IS NULL OR sbp_max IS NULL", conn)
print("Missing systolic BP values:", sbp_null.iloc[0]['COUNT(*)'])

#check missing diastolic blood pressure values
dbp_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE dbp_mean IS NULL OR dbp_min IS NULL OR dbp_max IS NULL", conn)
print("Missing diastolic BP values:", dbp_null.iloc[0]['COUNT(*)'])

#check missing temperature values
temperature_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE temperature_mean IS NULL OR temperature_min IS NULL", conn)
print("Missing temperature values:", temperature_null.iloc[0]['COUNT(*)'])

#check missing weight values
weight_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE weight_mean IS NULL", conn)
print("Missing weight values:", weight_null.iloc[0]['COUNT(*)'])

#check missing GCS values
gcs_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE gcs_min IS NULL", conn)
print("Missing GCS values:", gcs_null.iloc[0]['COUNT(*)'])

#check missing platelet values
platelets_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE platelets_mean IS NULL", conn)
print("Missing platelet values:", platelets_null.iloc[0]['COUNT(*)'])

#check missing haemoglobin values
haemoglobin_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE haemoglobin_mean IS NULL", conn)
print("Missing haemoglobin values:", haemoglobin_null.iloc[0]['COUNT(*)'])

#check missing potassium values
potassium_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE potassium_mean IS NULL", conn)
print("Missing potassium values:", potassium_null.iloc[0]['COUNT(*)'])

#check missing sodium values
sodium_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE sodium_mean IS NULL", conn)
print("Missing sodium values:", sodium_null.iloc[0]['COUNT(*)'])

#check missing white blood cell values
wbc_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE wbc_mean IS NULL", conn)
print("Missing white blood cell values:", wbc_null.iloc[0]['COUNT(*)'])

#check missing red blood cell values
rbc_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE rbc_mean IS NULL", conn)
print("Missing red blood cell values:", rbc_null.iloc[0]['COUNT(*)'])

#check missing hypertension values
hypertension_null = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE hypertension_count IS NULL", conn)
print("Missing hypertension values:", hypertension_null.iloc[0]['COUNT(*)'])

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
hypertension_negative = pd.read_sql_query("SELECT COUNT(*) FROM Final_Dataset WHERE hypertension_count < 0 OR hypertension_count >1", conn)
print("Invalid hypertension values:", hypertension_negative.iloc[0]['COUNT(*)'])

#manually checking whether aki labels are correct using a small sample
#pick 10 patients with aki and 10 without aki
aki_sample = pd.read_sql_query("SELECT subject_id, hadm_id, stay_id, AKI FROM Final_Dataset WHERE AKI = 1 ORDER BY RANDOM() LIMIT 10", conn)
no_aki_sample = pd.read_sql_query("SELECT subject_id, hadm_id, stay_id, AKI FROM Final_Dataset WHERE AKI = 0 ORDER BY RANDOM() LIMIT 10", conn)
sample = pd.concat([aki_sample, no_aki_sample])
print("Validation AKI counts:")
print(sample["AKI"].value_counts())
stay_ids = sample["stay_id"].tolist()
#get creatinine values for patients
sample_creatinine = pd.read_sql_query("SELECT f.subject_id, f.hadm_id, f.stay_id, f.AKI, c.charttime, c.intime, c.valuenum, c.valueuom FROM Final_Dataset f JOIN ICH_Creatinine c ON f.subject_id = c.subject_id AND f.hadm_id = c.hadm_id WHERE c.itemid IN (50912) AND c.valuenum IS NOT NULL ORDER BY f.stay_id, c.charttime", conn)
#work out how many hours after icu admission creatinine reading was
sample_creatinine["charttime"] = pd.to_datetime(sample_creatinine["charttime"])
sample_creatinine["intime"] = pd.to_datetime(sample_creatinine["intime"])
sample_creatinine["hours_from_icu"] = (sample_creatinine["charttime"] - sample_creatinine["intime"]).dt.total_seconds()/3600
sample_creatinine = sample_creatinine[sample_creatinine["stay_id"].isin(stay_ids)]

#only keep readings from up to seven days before icu admission 
sample_creatinine = sample_creatinine[sample_creatinine["hours_from_icu"] >= - (7*24)]
aki_check = []
for stay_id in sample_creatinine["stay_id"].unique():
    patient = sample_creatinine[sample_creatinine["stay_id"] == stay_id].copy()
    patient = patient.sort_values("charttime")
    aki_label = patient["AKI"].iloc[0]
    aki_found = 0
    first_value = None
    second_value = None
    creatinine_values = list(patient["valuenum"])
    times = list(patient["charttime"])
    hours_from_icu = list(patient["hours_from_icu"])
    for i in range(len(creatinine_values)):
        for j in range(i + 1, len(creatinine_values)):
            hours_between = (times[j] - times[i]).total_seconds()/3600
            #only count aki if second reading after 24 hours
            if hours_from_icu[j] > 24:
                #0.3 rise within 48 hours
                if hours_between <= 48 and creatinine_values[j] - creatinine_values[i] >= 0.3:
                    aki_found = 1
                    first_value = creatinine_values[i]
                    second_value = creatinine_values[j]
                #1.5 times rise within 7 days
                if hours_between <= 7*24 and creatinine_values[j] >= 1.5*creatinine_values[i]:
                    aki_found = 1
                    first_value = creatinine_values[i]
                    second_value = creatinine_values[j]
    if aki_label == aki_found:
        check = "Correct label"
    else:
        check = "Potentially incorrect"
    aki_check.append({"stay_id": stay_id, "AKI_label": aki_label, "AKI_found": aki_found, "first_value": first_value, "second_value": second_value, "check": check })
aki_check = pd.DataFrame(aki_check)
print(aki_check['Check'].value_counts())
