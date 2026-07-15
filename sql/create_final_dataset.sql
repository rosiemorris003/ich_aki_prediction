--combining all of the demographic, lab and chart events into final dataset
CREATE TABLE Final_Dataset AS
SELECT
    a.subject_id,
    a.hadm_id,
    f.stay_id,
    p.anchor_age,
    p.gender,
    h.hypertension_count,
    hr.heart_rate_mean,
    hr.heart_rate_max,
    s.sbp_mean,
    s.sbp_min,
    s.sbp_max,
    d.dbp_mean,
    d.dbp_min,
    d.dbp_max,
    t.temperature_mean,
    t.temperature_min,
    w.weight_mean,
    g.gcs_min,
    pl.platelets_mean,
    hb.haemoglobin_mean,
    po.potassium_mean,
    so.sodium_mean,
    wb.wbc_mean,
    rb.rbc_mean,
    a.AKI

FROM ICH_AKI_label a
JOIN ICH_First_ICU_Stay f ON a.subject_id = f.subject_id AND a.hadm_id = f.hadm_id
JOIN patients p ON a.subject_id = p.subject_id
LEFT JOIN ICH_Hypertension h ON a.subject_id = h.subject_id AND a.hadm_id = h.hadm_id
LEFT JOIN heart_rate hr ON a.subject_id = hr.subject_id AND a.hadm_id = hr.hadm_id
LEFT JOIN sbp s ON a.subject_id = s.subject_id AND a.hadm_id = s.hadm_id
LEFT JOIN dbp d ON a.subject_id = d.subject_id AND a.hadm_id = d.hadm_id
LEFT JOIN temperature t ON a.subject_id = t.subject_id AND a.hadm_id = t.hadm_id
LEFT JOIN weight w ON a.subject_id = w.subject_id AND a.hadm_id = w.hadm_id
LEFT JOIN gcs g ON a.subject_id = g.subject_id AND a.hadm_id = g.hadm_id
LEFT JOIN platelets pl ON a.subject_id = pl.subject_id AND a.hadm_id = pl.hadm_id
LEFT JOIN haemoglobin hb ON a.subject_id = hb.subject_id AND a.hadm_id = hb.hadm_id
LEFT JOIN potassium po ON a.subject_id = po.subject_id AND a.hadm_id = po.hadm_id
LEFT JOIN sodium so ON a.subject_id = so.subject_id AND a.hadm_id = so.hadm_id
LEFT JOIN wbc wb ON a.subject_id = wb.subject_id AND a.hadm_id = wb.hadm_id
LEFT JOIN rbc rb ON a.subject_id = rb.subject_id AND a.hadm_id = rb.hadm_id
