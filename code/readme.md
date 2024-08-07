
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
8. FinnGen proteomics and endpoints data.

# Other files called in codes that were downloaded from UKB or created by our team (`files/` directory):

1. UKB icd9 coding scheme (icd9_coding87.tsv) and age-related disease icd9 diagnosis codes (icd9_codings.R).
2. UKB icd10 coding scheme (icd10_coding19.tsv) and cancer / age-related disease icd10 diagnosis codes (icd10_codings.R; icd10_cancer_codings.R).
3. Uniprot IDs for protAge proteins (ProtAge_proteins_2023-12-29.csv).
4. Files with CpGs from existing DNAm clocks (DunedinPACE_genes.csv, Horvath_clock_cpgs.csv, Levine_PhenoAge_cpgs.csv).
5. Files with proteins from existing proteomic clocks (johnson_2020_clock_proteins.csv, Lehallier_2023_aging_proteins.csv, lehallier_nature_med_2019_clock.csv, lehallier_nature_med_2019_nomenclature.csv).
6. Names of coded Olink variables (olink_names_oct_30_2023.csv).
7. Primary care read codes to ICD lookup tables (primarycare_codings folder).

