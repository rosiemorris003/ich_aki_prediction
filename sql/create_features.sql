-- create summary tables for lab features for first 24 hours of submission including only realistic values
CREATE TABLE platelets AS
SELECT subject_id, hadm_id, AVG(valuenum) AS platelets_mean, MIN(valuenum) AS platelets_min, MAX(valuenum) AS platelets_max
FROM ICH_LabEvents_24h 
WHERE itemid = 51265 AND valuenum BETWEEN 1 AND 1500 GROUP BY subject_id, hadm_id;

CREATE TABLE haemoglobin AS
SELECT subject_id, hadm_id, AVG(valuenum) AS haemoglobin_mean, MIN(valuenum) AS haemoglobin_min, MAX(valuenum) AS haemoglobin_max
FROM ICH_LabEvents_24h 
WHERE itemid = 51222 AND valuenum BETWEEN 2 AND 25 GROUP BY subject_id, hadm_id;

CREATE TABLE haematocrit AS
SELECT subject_id, hadm_id, AVG(valuenum) AS haematocrit_mean, MIN(valuenum) AS haematocrit_min, MAX(valuenum) AS haematocrit_max
FROM ICH_LabEvents_24h 
WHERE itemid = 51221 AND valuenum BETWEEN 5 AND 70 GROUP BY subject_id, hadm_id;

CREATE TABLE potassium AS
SELECT subject_id, hadm_id, AVG(valuenum) AS potassium_mean, MIN(valuenum) AS potassium_min, MAX(valuenum) AS potassium_max
FROM ICH_LabEvents_24h 
WHERE itemid = 50971 AND valuenum BETWEEN 1 AND 10 GROUP BY subject_id, hadm_id;

CREATE TABLE sodium AS
SELECT subject_id, hadm_id,AVG(valuenum) AS sodium_mean, MIN(valuenum) AS sodium_min, MAX(valuenum) AS sodium_max
FROM ICH_LabEvents_24h 
WHERE itemid = 50983 AND valuenum BETWEEN 100 AND 180 GROUP BY subject_id, hadm_id;

CREATE TABLE wbc AS
SELECT subject_id, hadm_id, AVG(valuenum) AS wbc_mean, MIN(valuenum) AS wbc_min, MAX(valuenum) AS wbc_max
FROM ICH_LabEvents_24h 
WHERE itemid = 51301 AND valuenum BETWEEN 0.1 AND 300 GROUP BY subject_id, hadm_id;

CREATE TABLE rbc AS
SELECT subject_id, hadm_id, AVG(valuenum) AS rbc_mean, MIN(valuenum) AS rbc_min, MAX(valuenum) AS rbc_max
FROM ICH_LabEvents_24h 
WHERE itemid = 51279 AND valuenum BETWEEN 1 AND 8 GROUP BY subject_id, hadm_id;

-- create summary tables for vital signs from first 24 hours of icu admission

CREATE TABLE heart_rate AS
SELECT subject_id, hadm_id, AVG(valuenum) AS heart_rate_mean, MIN(valuenum) AS heart_rate_min, MAX(valuenum) AS heart_rate_max
FROM ICH_ChartEvents_24h 
WHERE itemid = 220045 AND valuenum BETWEEN 20 AND 250 GROUP BY subject_id, hadm_id;

CREATE TABLE sbp AS
SELECT subject_id, hadm_id, AVG(valuenum) AS sbp_mean, MIN(valuenum) AS sbp_min, MAX(valuenum) AS sbp_max
FROM ICH_ChartEvents_24h 
WHERE itemid = 220179 AND valuenum BETWEEM 30 AND 300 GROUP BY subject_id, hadm_id;

CREATE TABLE dbp AS
SELECT subject_id, hadm_id, AVG(valuenum) AS dbp_mean, MIN(valuenum) AS dbp_min, MAX(valuenum) AS dbp_max
FROM ICH_ChartEvents_24h 
WHERE itemid = 220180 AND valuenum BETWEEN 10 AND 200 GROUP BY subject_id, hadm_id;

--convert farenheit temp to celsius and remove unrealistic values
CREATE TABLE temperature_clean AS
SELECT subject_id, hadm_id, charttime, valuenum AS temp_c
FROM ICH_ChartEvents_24h 
WHERE itemid = 223762 AND valuenum BETWEEN 30 AND 45;

INSERT INTO temperature_clean
SELECT subject_id, hadm_id, charttime, (valuenum - 32) * 5.0 / 9.0 AS temp_c
FROM ICH_ChartEvents_24h 
WHERE itemid = 223761 AND valuenum IS NOT NULL;

-- create summary temperature table
CREATE TABLE temperature AS
SELECT subject_id, hadm_id, AVG(temp_c) AS temperature_mean, MIN(temp_c) AS temperature_min, MAX(temp_c) AS temperature_max
FROM temperature_clean 
WHERE temp_c BETWEEN 30 AND 45 GROUP BY subject_id, hadm_id;

CREATE TABLE weight AS
SELECT subject_id, hadm_id, AVG(valuenum) AS weight_mean
FROM ICH_ChartEvents_24h 
WHERE itemid = 224639 AND valuenum BETWEEN 30 AND 300 GROUP BY subject_id, hadm_id;


--calculate the total gcs score from the eye, verbal and motor components
CREATE TABLE gcs_parts AS
SELECT subject_id, hadm_id, charttime, SUM(valuenum) AS gcs_total
FROM ICH_ChartEvents_24h 
WHERE itemid = 220739 OR itemid = 223900 OR itemid = 223901 GROUP BY subject_id, hadm_id, charttime;

--create min gcs feature from the first 24 hours
CREATE TABLE gcs AS
SELECT subject_id, hadm_id, MIN(gcs_total) AS gcs_min
FROM gcs_parts 
WHERE gcs_total BETWEEN 3 AND 15 GROUP BY subject_id, hadm_id;
