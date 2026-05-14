CREATE TABLE ICH_LabEvents_24h AS
SELECT
    f.subject_id,
    f.hadm_id,
    f.stay_id,
    l.itemid,
    l.charttime,
    l.valuenum,
    l.valueuom
FROM ICH_First_ICU_Stay f
JOIN ICH_AKI_label a
    ON f.subject_id = a.subject_id AND f.hadm_id = a.hadm_id
JOIN LabEvents l
    ON f.subject_id = l.subject_id AND f.hadm_id = l.hadm_id
WHERE l.charttime >= f.intime AND l.charttime < datetime(f.intime, '+24 hours') AND l.valuenum IS NOT NULL;

CREATE TABLE ICH_ChartEvents_24h AS
SELECT
    f.subject_id,
    f.hadm_id,
    f.stay_id,
    c.itemid,
    c.charttime,
    c.valuenum,
    c.valueuom
FROM ICH_First_ICU_Stay f
JOIN ICH_AKI_label a
    ON f.subject_id = a.subject_id AND f.hadm_id = a.hadm_id
JOIN ChartEvents c
    ON f.subject_id = c.subject_id AND f.hadm_id = c.hadm_id AND f.stay_id = c.stay_id
WHERE c.charttime >= f.intime AND c.charttime < datetime(f.intime, '+24 hours') AND c.valuenum IS NOT NULL;