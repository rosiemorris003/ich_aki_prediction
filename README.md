# ICH  AKI prediction
This project looks at whether clinical data that is collected in the first 24 hours of an ICU stay can be used to predict which intracerebral haemorrhage (ICH) patients will develop acute kidney injury (AKI) later on. It uses data from MIMIC-IV and compares seven different machine learning models.

## Dataset
MIMIC-IV was accessed through PhysioNet. The final dataset included 2496 adult ICH patients, with 719 developing AKI after the first 24 hours of ICU admission. As MIMIC-IV is a restricted access dataset, the raw patient data is not included in the repository.

## Predictors
The models used 21 features taken from the first 24 hours in the ICU. These included age, gender, hypertension, weight, heart rate, blood pressure, temperature, GCS and blood test results such as haemoglobin, platelets, potassium, sodium, WBC and RBC. 

## Models
The seven machine learning models used were: logistic regression, XGBoost, CatBoost, LightGBM, random forest, TabPFN and an artificial neural network (ANN). The models were tested using their original class distribution, random undersampling and where appropriate, hyperparameter tuning. 

## Workflow
The project followed a step by step process which started with the cohort creation and data checks, followed by feature preparation and reduction. The models were then trained, tested with random undersampling and tuning, calibration and interpreted using SHAP, odds ratios and feature removal analysis. 

## Evaluation 
The models were compared using accuracy, ROC-AUC, PR-AUC, F1 score, precision, recall and specificity. Brier scores and calibration curves were also used to check how reliable the probabilities were, as well as the bootstrap 95% confidence intervals being calculated for the ROC-AUC results. 

## Data access
As the data set is restricted, anyone wanting to reproduce the analysis will need to obtain their own approves access to MIMIC-IV and follow the PhysioNet data use requirements. 

## Repository structure
The repository is organised into separate files for the main stages of the project.
sql/ - contains the SQL scripts that are used to identify the ICH cohort, extract clinical measurements and create the tables needed for the analysis.
aki_labels.py - creates and checks the AKI labels using creatinine measurements 
raw_data.py - loads the required MIMIC-IV files into a local SQLite database, selecting the relevant columns and clinical item IDs. The large event files are processed in chunks to reduce memory usage
validation_checks.py - checks the final dataset for issues such as missing values, duplicates and unrealistic values
multicollinearity.py - carries out the correlation and VIF analysis used in feature reduction 
models.py - contains the model training, random undersampling, hyperparameter tuning, evaluation and calibration 
feature_removals.py - rerunes the seven standard models after selected predictors or groups of predictors are removed so their performance can be compared with the full feature set
.gitignore - prevents restricted data and unnecessary files from being uploaded to the repository

## How to run the project 
1. Execute create_raw_tables.sql in SQLite to set up the initial database tables 
2. Run raw_data.py to load the required MIMIC-IV data into the SQLite database
3. Execute create_ich_cohort.sql in SQLite to identify the ICH cohort and extract the creatinine measurements
4. Run AKI_labels.py to generate the AKI outcome labels 
5. Execute extract_24h_events.sql in SQLite to extract laboratory and chart measurements from the first 24 hours 
6. Execute create_features.sql in SQLite to create the predictor variables
7. Execute create_final_dataset.sql in SQLite to combine the predictors and AKI labels into the final modelling dataset
8. Run validation_checks.py to check the final dataset
9. Run multicollinearity.py to carry out the correlation and VIF analysis
10. Run models.py for model development and evaluation
11. Run feature_removals.py for the feature removal analysis 
MIMIC-IV is not included in this repository, so anyone reproducing the project will need their own approved access and will need to update the local data and SQLite database paths

## Requirements
The project was developed in Python and uses SQLite for data storage and querying.
The main Python libraries used include:
- pandas
- NumPy
- scikit-learn
- imbalanced-learn
- XGBoost 
- CatBoost
- LightGBM
- TensorFlow
- SciKeras
- TabPFN
- SHAP
- statsmodels
- matplotlib

## Reproducibility
A fixed random seed of 42 was used where supported to make the train-test split and model results more reproducible. Data preprocessing was fitted using the training data only as was the random undersampling, so that the test set kept the original class distribution.

## Main findings
The models achieved moderate predictive performance, with no single model clearly outperforming the others. 
The main findings were:
- The ANN achieved the highest standard ROC-AUC of 0.736
- Random undersampling generally improved recall but reduced specificity
- The stronger models had substantially overallping 95% ROC-AUC confidence intervals 
- Minimum GCS was the most consistently important predictor across the feature interpretation and removal analyses. 
The results suggest that there is useful predictive information within the first 24 hours of ICU admission but further validation would be needed before the models could be considered for clinical use. 

## Limitations
 The study used retrospective data from a single database, so the results may not generalise to other hospitals. AKI was identified using serum creatinine only, as urine output data was not considered relaible enough for the analysis. The models were also evaluated on an internal test and have not yet been externally validation. For these reasons, the models should be treated as research findings rather than tools that are ready for clinical use. 

 