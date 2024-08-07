library(Hmisc)
library(readr)


# import list of ICD-10 codes from UKB website: https://biobank.ctsu.ox.ac.uk/crystal/coding.cgi?id=19&nl=1 
ICD10list3d <- read_delim("/Users/aargenti/Documents/Broad_UKB/code/icd10_coding19.tsv", 
                          "\t", 
                          escape_double = FALSE, 
                          trim_ws = TRUE)

# make character vector with just ICD codes
ICD10list3d <- as.vector(ICD10list3d$coding)

### defining chronic diseases by ICD 10 codes --------------------------------------------------

# IBD (K50 and K51)
prefix <- "K"
suffix <- seq(50,51)
seq <- paste0(prefix, suffix)

prefix <- "K"
suffix <- seq(500,519)
seq2 <- paste0(prefix, suffix)

IBD <- c(seq, seq2)
IBD_icd10 <- ICD10list3d[which(ICD10list3d %in% IBD)]


# Endometriosis (N80)
prefix <- "N"
suffix <- seq(800,809)
seq <- paste0(prefix, suffix)

endometriosis <- c("N80", seq)
endometriosis_icd10 <- ICD10list3d[which(ICD10list3d %in% endometriosis)]


# type II diabetes (E11)
prefix <- "E"
suffix <- seq(110,119)
seq <- paste0(prefix, suffix)

diabetes <- c("E11", seq)
t2diabetes_icd10 <- ICD10list3d[which(ICD10list3d %in% diabetes)]


# ischemic heart diseases (I20–I25)
prefix <- "I"
suffix <- seq(20,25)
seq <- paste0(prefix, suffix)

prefix <- "I"
suffix <- seq(200,259)
seq2 <- paste0(prefix, suffix)

cardio <- c(seq, seq2)
IHD_icd10 <- ICD10list3d[which(ICD10list3d %in% cardio)]

# ischemic stroke (I63-I64)
prefix <- "I"
suffix <- seq(640,649)
seq <- paste0(prefix, suffix)

ischemic_stroke <- 
  c("I63",
    "I630",
    "I631",
    "I632",
    "I633",
    "I634",
    "I635",
    "I636",
    "I638",
    "I639",
    "I64",
    seq)
ischemic_stroke_icd10 <- ICD10list3d[which(ICD10list3d %in% ischemic_stroke)]

# Subarachnoid hemorrhage (I60)
prefix <- "I"
suffix <- seq(600,609)
seq <- paste0(prefix, suffix)

SH_stroke <- c("I60", seq)
SH_stroke_icd10 <- ICD10list3d[which(ICD10list3d %in% SH_stroke)]

# Intracerebral hemorrhage (I61)
prefix <- "I"
suffix <- seq(610,619)
seq <- paste0(prefix, suffix)

IH_stroke <- c("I61", seq)
IH_stroke_icd10 <- ICD10list3d[which(ICD10list3d %in% IH_stroke)]

# all stroke
all_stroke_icd10 <- ICD10list3d[which(ICD10list3d %in% c(ischemic_stroke, SH_stroke, IH_stroke))]


# COPD and emphysema (J43–J44)
prefix <- "J"
suffix <- seq(43,44)
seq <- paste0(prefix, suffix)

prefix <- "J"
suffix <- seq(430,449)
seq2 <- paste0(prefix, suffix)

emphysema_COPD <- c(seq, seq2)
emphysema_COPD_icd10 <- ICD10list3d[which(ICD10list3d %in% emphysema_COPD)]

# COPD (J44)
prefix <- "J"
suffix <- seq(440,449)
seq <- paste0(prefix, suffix)

COPD <- c("J44", seq)
COPD_icd10 <- ICD10list3d[which(ICD10list3d %in% COPD)]


# Emphysema (J43)
prefix <- "J"
suffix <- seq(430,439)
seq <- paste0(prefix, suffix)

emphysema <- c("J43", seq)
emphysema_icd10 <- ICD10list3d[which(ICD10list3d %in% emphysema)]



# chronic liver disease (K70, K73-K74, K75.8, K76.0)
prefix <- "K"
suffix <- seq(73,74)
seq <- paste0(prefix, suffix)

prefix <- "K"
suffix <- seq(700,709)
seq2 <- paste0(prefix, suffix)

prefix <- "K"
suffix <- seq(730,749)
seq3 <- paste0(prefix, suffix)

liver <- c("K70", 
           seq, 
           seq2, 
           seq3, 
           "K758",
           "K760")
liver_icd10 <- ICD10list3d[which(ICD10list3d %in% liver)]


# chronic kidney disease (N18)
prefix <- "N"
suffix <- seq(180,189)
seq <- paste0(prefix, suffix)

kidney <- c("N18", seq)
kidney_icd10 <- ICD10list3d[which(ICD10list3d %in% kidney)]


# all-cause dementia (A81.0, F00-F03, F05.1, F10.6, G30-G31, I67.3)
prefix <- "F"
suffix <- sprintf('%0.3d', 0:9)
seq <- paste0(prefix, suffix)

prefix <- "F0"
suffix <- sprintf('%0.d', 10:39)
seq2 <- paste0(prefix, suffix)

prefix <- "G"
suffix <- seq(300,309)
seq3 <- paste0(prefix, suffix)

prefix <- "G"
suffix <- seq(310,319)
seq4 <- paste0(prefix, suffix)

all_dementia <- c("A810", "F00", "F01", "F02", "F03", "G30", "G31", seq, seq2, seq3, seq4, "F051", "F106", "I673")
all_dementia_icd10 <- ICD10list3d[which(ICD10list3d %in% all_dementia)]

# vascular dementia (F01, I67.3)
prefix <- "F0"
suffix <- sprintf('%0.d', 10:19)
seq <- paste0(prefix, suffix)

vasc_dementia <- c("F01", seq, "I673")
vasc_dementia_icd10 <- ICD10list3d[which(ICD10list3d %in% vasc_dementia)]

# Alzheimer's (F00, G30)
prefix <- "F"
suffix <- sprintf('%0.3d', 0:9)
seq <- paste0(prefix, suffix)

prefix <- "G"
suffix <- seq(300,309)
seq2 <- paste0(prefix, suffix)

alzheimers <- c("F00", seq, "G30", seq2)
alzheimers_icd10 <- ICD10list3d[which(ICD10list3d %in% alzheimers)]

# Parkinsons (G20-G22)
prefix <- "G"
suffix <- seq(200,229)
seq <- paste0(prefix, suffix)

parkinsons <- c("G20", "G21", "G22", seq)
parkinsons_icd10 <- ICD10list3d[which(ICD10list3d %in% parkinsons)]


# rheumatoid arthritis (M05-06)
prefix <- "M05"
suffix <- seq(0,9)
seq <- paste0(prefix, suffix)

prefix <- "M06"
suffix <- seq(0,9)
seq2 <- paste0(prefix, suffix)

rheumatoid <- c("M05",
                seq,
                "M06",
                seq2)
rheumatoid_icd10 <- ICD10list3d[which(ICD10list3d %in% rheumatoid)]


# macular degeneration (H35.3)
macular <- "H353"
macular_degen_icd10 <- ICD10list3d[which(ICD10list3d %in% macular)]


# osteoporosis (M80-M81)
prefix <- "M8"
suffix <- sprintf('%0.2d', 00:19)
seq <- paste0(prefix, suffix)

osteoporosis <- c("M80",
                  "M81",
                  seq)
osteoporosis_icd10 <- ICD10list3d[which(ICD10list3d %in% osteoporosis)]


# osteoarthritis (M15-M19)
prefix <- "M"
suffix <- seq(15,19)
seq <- paste0(prefix, suffix)

prefix <- "M1"
suffix <- sprintf('%0.2d', 50:99)
seq2 <- paste0(prefix, suffix)

osteoarthritis <- c(seq, seq2)
osteoarthritis_icd10 <- ICD10list3d[which(ICD10list3d %in% osteoarthritis)]


## clinical endophenotypes

# hypertension (I10–I15)
prefix <- "I"
suffix <- seq(10,15)
seq <- paste0(prefix, suffix)

prefix <- "I"
suffix <- seq(100,159)
seq2 <- paste0(prefix, suffix)

hypertension <- c(seq, seq2)
hypertension_icd10 <- ICD10list3d[which(ICD10list3d %in% hypertension)]

# obesity (E66)
prefix <- "E66"
suffix <- sprintf('%0.1d', 0:9)
seq <- paste0(prefix, suffix)

obesity <- c("E66", seq)
obesity_icd10 <- ICD10list3d[which(ICD10list3d %in% obesity)]

# dyslipidemia (E78)
prefix <- "E78"
suffix <- sprintf('%0.1d', 0:9)
seq <- paste0(prefix, suffix)

dyslipidemia <- c("E78", seq)
dyslipidemia_icd10 <- ICD10list3d[which(ICD10list3d %in% dyslipidemia)]


# create list of all NCD codes
icd10_include <- list(
  IBD_icd10,
  endometriosis_icd10,
  t2diabetes_icd10,
  IHD_icd10,
  ischemic_stroke_icd10,
  IH_stroke_icd10,
  SH_stroke_icd10,
  all_stroke_icd10,
  emphysema_COPD_icd10,
  COPD_icd10,
  emphysema_icd10,
  liver_icd10, 
  kidney_icd10,
  all_dementia_icd10,
  vasc_dementia_icd10,
  alzheimers_icd10,
  parkinsons_icd10,
  rheumatoid_icd10,
  macular_degen_icd10,
  osteoporosis_icd10,
  osteoarthritis_icd10,
  hypertension_icd10,
  obesity_icd10,
  dyslipidemia_icd10
)

names(icd10_include) <- c(
  Cs(
    IBD_icd10,
    endometriosis_icd10,
    t2diabetes_icd10,
    IHD_icd10,
    ischemic_stroke_icd10,
    IH_stroke_icd10,
    SH_stroke_icd10,
    all_stroke_icd10,
    emphysema_COPD_icd10,
    COPD_icd10,
    emphysema_icd10,
    liver_icd10, 
    kidney_icd10,
    all_dementia_icd10,
    vasc_dementia_icd10,
    alzheimers_icd10,
    parkinsons_icd10,
    rheumatoid_icd10,
    macular_degen_icd10,
    osteoporosis_icd10,
    osteoarthritis_icd10,
    hypertension_icd10,
    obesity_icd10,
    dyslipidemia_icd10
  )
)