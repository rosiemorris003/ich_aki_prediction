--store the patient demographic info 
CREATE TABLE patients (
    subject_id INTEGER NOT NULL PRIMARY KEY,
    anchor_age INTEGER,
    gender TEXT
);
--keep icu admission info
CREATE TABLE icu_stays (
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER NOT NULL,
    stay_id INTEGER NOT NULL PRIMARY KEY,
    intime TEXT NOT NULL,
    los REAL
);
--icd diagnosis codes and their descriptions
CREATE TABLE ICD_CODES (
    icd_code TEXT NOT NULL,
    icd_version INTEGER NOT NULL,
    long_title TEXT,
    PRIMARY KEY (icd_code, icd_version)
);
--diagnosis codes for each patient
CREATE TABLE DIAGNOSES (
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER NOT NULL,
    seq_num INTEGER NOT NULL,
    icd_code TEXT NOT NULL,
    icd_version INTEGER NOT NULL,
    PRIMARY KEY (subject_id, hadm_id, seq_num)
);
--describes lab test items and their codes
CREATE TABLE LabItems (
    itemid INTEGER NOT NULL PRIMARY KEY,
    label TEXT,
    fluid TEXT,
    category TEXT
);
--describes chart test items and their codes
CREATE TABLE ChartItems (
    itemid INTEGER NOT NULL PRIMARY KEY,
    label TEXT,
    abbreviation TEXT,
    linksto TEXT,
    category TEXT,
    unitname TEXT,
    param_type TEXT,
    lownormalvalue REAL,
    highnormalvalue REAL
);
--store icu chart observations including vitals
CREATE TABLE ChartEvents (
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER,
    stay_id INTEGER NOT NULL,
    itemid INTEGER NOT NULL,
    charttime TEXT NOT NULL,
    valuenum REAL,
    valueuom TEXT
);
--store lab test results recorded during hospital admission
CREATE TABLE LabEvents (
    labevent_id INTEGER NOT NULL PRIMARY KEY,
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER NOT NULL,
    specimen_id INTEGER,
    itemid INTEGER NOT NULL,
    charttime TEXT NOT NULL,
    value TEXT,
    valuenum REAL,
    valueuom TEXT,
    ref_range_lower REAL,
    ref_range_upper REAL,
    flag TEXT,
    comments TEXT
);
