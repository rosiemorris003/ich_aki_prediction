CREATE TABLE ICH_Patients AS
SELECT DISTINCT
    d.subject_id,
    d.hadm_id
FROM DIAGNOSES d
JOIN ICD_CODES c
    ON d.icd_code = c.icd_code AND d.icd_version = c.icd_version
WHERE 
    (d.icd_version = 9 AND d.icd_code = '431') OR (d.icd_version = 10 AND d.icd_code LIKE 'I61%');
CREATE TABLE ICH_ICU_Stays AS
SELECT
    i.subject_id,
    i.hadm_id,
    i.stay_id,
    i.intime
FROM icu_stays i
JOIN ICH_Patients p
    ON i.subject_id = p.subject_id AND i.hadm_id = p.hadm_id;
 
CREATE TABLE ICH_First_ICU_Stay AS
SELECT
    i.subject_id,
    i.hadm_id,
    i.stay_id,
    i.intime
FROM ICH_ICU_Stays i
WHERE i.intime = (
    SELECT MIN(earliest.intime)
    FROM ICH_ICU_Stays earliest
    WHERE earliest.subject_id = i.subject_id
);
CREATE TABLE ICH_Hypertension AS
SELECT
    f.subject_id,
    f.hadm_id,
    COUNT(d.icd_code) AS hypertension_count
FROM ICH_First_ICU_Stay f
LEFT JOIN DIAGNOSES d
    ON f.subject_id = d.subject_id AND f.hadm_id = d.hadm_id AND (d.icd_code = '4010' OR d.icd_code = '4011' OR d.icd_code = '4019' OR d.icd_code = 'I10')
GROUP BY f.subject_id, f.hadm_id;

CREATE TABLE ICH_Creatinine AS
SELECT
    f.subject_id,
    f.hadm_id,
    f.stay_id,
    f.intime,
    l.itemid,
    l.charttime,
    l.valuenum,
    l.valueuom
FROM ICH_First_ICU_Stay f
JOIN LabEvents l
    ON f.subject_id = l.subject_id AND f.hadm_id = l.hadm_id
WHERE l.itemid = 51081 OR l.itemid = 50912;

CREATE TABLE ICH_AKI_label_all (
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER NOT NULL,
    AKI_first_24h INTEGER NOT NULL,
    AKI_after_24h INTEGER NOT NULL,
    PRIMARY KEY (subject_id, hadm_id)
);

CREATE TABLE ICH_AKI_label (
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER NOT NULL,
    AKI INTEGER NOT NULL,
    PRIMARY KEY (subject_id, hadm_id)
);