library(Hmisc)

# import list of ICD-10 codes from UKB website: https://biobank.ctsu.ox.ac.uk/crystal/coding.cgi?id=19&nl=1 
ICD10list3d <- read_delim("/Users/aargenti/Documents/proteomic_age/data/icd10_coding19.tsv", 
                          "\t", 
                          escape_double = FALSE, 
                          trim_ws = TRUE)

# make character vector with just ICD codes
ICD10list3d <- as.vector(ICD10list3d$coding)

### codes

# breast cancer
code <- "C50"
breast_cancer_icd10 <- ICD10list3d[which(startsWith(ICD10list3d, code))]

# lung cancer
codes <- c("C33", "C34")
lung_cancer_icd10 <- as.vector(
  unlist(
    sapply(codes, function(x) ICD10list3d[which(startsWith(ICD10list3d, x))])
  )
)

# prostate cancer
code <- "C61"
prostate_cancer_icd10 <- ICD10list3d[which(startsWith(ICD10list3d, code))]

# colorectal cancer
codes <- c("C18", "C19", "C20")
colorectal_cancer_icd10 <- as.vector(
  unlist(
    sapply(codes, function(x) ICD10list3d[which(startsWith(ICD10list3d, x))])
  )
)

# skin cancer
code <- "C43"
skin_cancer_icd10 <- ICD10list3d[which(startsWith(ICD10list3d, code))]

# non-hodgkin lymphoma
codes <- c("C82", "C83", "C84", "C85", "C86")
nh_lymphoma_icd10 <- as.vector(
  unlist(
    sapply(codes, function(x) ICD10list3d[which(startsWith(ICD10list3d, x))])
  )
)


# pancreatic cancer
code <- "C25"
pancreatic_cancer_icd10 <- ICD10list3d[which(startsWith(ICD10list3d, code))]

# Kidney cancer
code <- "C64"
kidney_cancer_icd10 <- ICD10list3d[which(startsWith(ICD10list3d, code))]

# bladder cancer
code <- "C67"
bladder_cancer_icd10 <- ICD10list3d[which(startsWith(ICD10list3d, code))]

# oral
codes <- c("C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10","C11","C12", "C13", "C14")
oral_cancer_icd10 <- as.vector(
  unlist(
    sapply(codes, function(x) ICD10list3d[which(startsWith(ICD10list3d, x))])
  )
)

# uterus
codes <- c("C54", "C55")
uterus_cancer_icd10 <- as.vector(
  unlist(
    sapply(codes, function(x) ICD10list3d[which(startsWith(ICD10list3d, x))])
  )
)

# leukemia
codes <- c("C91", "C92", "C93", "C94", "C95")
leukemia_icd10 <- as.vector(
  unlist(
    sapply(codes, function(x) ICD10list3d[which(startsWith(ICD10list3d, x))])
  )
)

# esophageal
code <- "C15"
eso_cancer_icd10 <- ICD10list3d[which(startsWith(ICD10list3d, code))]

# ovarian
codes <- c("C56", "C57")
ovarian_cancer_icd10 <- as.vector(
  unlist(
    sapply(codes, function(x) ICD10list3d[which(startsWith(ICD10list3d, x))])
  )
)

# liver
code <- "C22"
liver_cancer_icd10 <- ICD10list3d[which(startsWith(ICD10list3d, code))]

# stomach
code <- "C16"
stomach_cancer_icd10 <- ICD10list3d[which(startsWith(ICD10list3d, code))]

# myeloma
code <- "C90"
myeloma_icd10 <- ICD10list3d[which(startsWith(ICD10list3d, code))]

# secondary malignant neoplasm
code <- "C79"
secondary_cancer_icd10 <- ICD10list3d[which(startsWith(ICD10list3d, code))]

# brain
codes <- c("C71", "C72")
brain_cancer_icd10 <- as.vector(
  unlist(
    sapply(codes, function(x) ICD10list3d[which(startsWith(ICD10list3d, x))])
  )
)

# thyroid
code <- "C73"
thyroid_cancer_icd10 <- ICD10list3d[which(startsWith(ICD10list3d, code))]

# cervical
code <- "C53"
cervical_cancer_icd10 <- ICD10list3d[which(startsWith(ICD10list3d, code))]

# Mesothelioma
code <- "C45"
mesothelioma_icd10 <- ICD10list3d[which(startsWith(ICD10list3d, code))]

# Testis
code <- "C62"
testicular_cancer_icd10 <- ICD10list3d[which(startsWith(ICD10list3d, code))]

# Hodgkin lymphoma
code <- "C81"
h_lymphoma_icd10 <- ICD10list3d[which(startsWith(ICD10list3d, code))]

# Larynx
code <- "C32"
larynx_cancer_icd10 <- ICD10list3d[which(startsWith(ICD10list3d, code))]

# Cancer of unknown primary (CUP)
code <- "C80"
CUP_cancer_icd10 <- ICD10list3d[which(startsWith(ICD10list3d, code))]


# create list of all NCD codes
icd10_cancer_include <- list(
  breast_cancer_icd10,
  lung_cancer_icd10,
  prostate_cancer_icd10,
  colorectal_cancer_icd10,
  skin_cancer_icd10,
  nh_lymphoma_icd10,
  pancreatic_cancer_icd10,
  kidney_cancer_icd10,
  bladder_cancer_icd10,
  oral_cancer_icd10,
  uterus_cancer_icd10,
  leukemia_icd10,
  eso_cancer_icd10,
  ovarian_cancer_icd10,
  liver_cancer_icd10,
  stomach_cancer_icd10,
  myeloma_icd10,
  secondary_cancer_icd10,
  brain_cancer_icd10,
  thyroid_cancer_icd10,
  cervical_cancer_icd10,
  mesothelioma_icd10,
  testicular_cancer_icd10,
  h_lymphoma_icd10,
  larynx_cancer_icd10,
  CUP_cancer_icd10
)

names(icd10_cancer_include) <- c(
  Cs(
    breast_cancer_icd10,
    lung_cancer_icd10,
    prostate_cancer_icd10,
    colorectal_cancer_icd10,
    skin_cancer_icd10,
    nh_lymphoma_icd10,
    pancreatic_cancer_icd10,
    kidney_cancer_icd10,
    bladder_cancer_icd10,
    oral_cancer_icd10,
    uterus_cancer_icd10,
    leukemia_icd10,
    eso_cancer_icd10,
    ovarian_cancer_icd10,
    liver_cancer_icd10,
    stomach_cancer_icd10,
    myeloma_icd10,
    secondary_cancer_icd10,
    brain_cancer_icd10,
    thyroid_cancer_icd10,
    cervical_cancer_icd10,
    mesothelioma_icd10,
    testicular_cancer_icd10,
    h_lymphoma_icd10,
    larynx_cancer_icd10,
    CUP_cancer_icd10
  )
)

