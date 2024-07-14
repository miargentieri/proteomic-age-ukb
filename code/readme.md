
July 14, 2024


### Code to run all data prep, analysis, and figure/table generation for the publication "Proteomic aging clock predicts mortality and risk of common age-related diseases in diverse populations".


# To run code, one will need:

1. A raw UK Biobank (UKB) dataset from the UKB showcase (was used earlier in analysis stages).
2. A raw UKB dataset from the UKB RAP (was used later in analysis stages to download updated UKB data fields.
3. UKB Olink data tables downloaded from the showcase.
4. Mortality, hospital inpatient (HES), and primary care data tables downloaded from the RAP.
5. Cancer registry tabular data downloaded from the UKB RAP.
6. Files mapping plate IDs and batch numbers to UKB Olink data.
7. Raw China Kadoorie Biobank (CKB) questionnaire, Olink, and endpoints data.


# Code files are run in the following order:

1. Import CKB raw questionnaire and endpoint data (CKD-data-import.Rmd).
2. Import CKB Olink and mortality data (CKB-olink-mort-import.Rmd).
3. Import UKB raw data (UKB-data-import.Rmd).
4. Clean and recode UKB data (UKB_data-recoding.Rmd).
5. Impute missing values and code derived variables in UKB data (UKB-data-imputation.Rmd).
6. Code granular age variable in UKB (UKB-granular-age.R).
7. Import UKB Olink data (UKB-import-olink.Rmd).
8. Code UKB mortality data (UKB-mortality-coding.Rmd).
9. Code UKB non-cancer incident disease outcomes (UKB-disease-diagnosis-coding.Rmd).
10. Code UKB cancer incident outcomes (UKB-cancer-diagnosis-coding.Rmd).
11. Script with machine learning functions (lgbm_functions.py) that is loaded when running the proteomic age model. Make sure this is saved in same directory as proteomic_age_model.py.
12. Run proteomic age model (proteomic_age_model.py). 
13. Run regression analyses, create figures and tables (figures_and_regressions.ipynb).


# Other files called in codes that were downloaded from UKB or created myself (other_files folder):

1. UKB icd9 coding scheme (icd9_coding87.tsv).
2. UKB icd10 coding scheme (icd10_coding19.tsv).
3. Uniprot IDs for protAge proteins (uniprot-230-proteins-to-check-jul-26-23.csv).
4. Files with CpGs from existing DNAm clocks (DunedinPACE_genes.csv, Horvath_clock_cpgs.csv, Levine_PhenoAge_cpgs.csv).
5. Files with proteins from existing proteomic clocks (johnson_2020_clock_proteins.csv, Lehallier_2023_aging_proteins.csv, lehallier_nature_med_2019_clock.csv, lehallier_nature_med_2019_nomenclature.csv).
6. Names of coded Olink variables (olink_names.csv).
7. Primary care read codes to ICD lookup tables (primarycare_codings folder).
