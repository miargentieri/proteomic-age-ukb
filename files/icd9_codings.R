library(Hmisc)
library(readr)

icd9list <- read_delim("/Users/aargenti/Documents/Broad_UKB/code/icd9_coding87.tsv", 
                       "\t", 
                       escape_double = FALSE, 
                       trim_ws = TRUE)

# IBD (555[0,1,2,9] and 556[9])
prefix <- "555"
suffix <- sprintf('%0.1d', 0:9)
seq <- paste0(prefix, suffix)

prefix <- "556"
suffix <- sprintf('%0.1d', 0:9)
seq2 <- paste0(prefix, suffix)

IBD <- c("555", "555", seq, seq2)
IBD_icd9 <- icd9list$coding[which(icd9list$coding %in% IBD)]

# Endometriosis: 617[0–9]
prefix <- "617"
suffix <- sprintf('%0.1d', 0:9)
seq <- paste0(prefix, suffix)

endometriosis <- c("617", seq)
endometriosis_icd9 <- icd9list$coding[which(icd9list$coding %in% endometriosis)]


# type II diabetes (E11)
prefix <- "250"
suffix <- sprintf('%0.1d', 0:9)
seq <- paste0(prefix, suffix)

list <- as.list(rep(1, length(seq)))
for (k in seq_along(seq)) {
  list[[k]] <- paste0(seq[k], sprintf('%0.1d', 0:9))
}

list <- unlist(list)

diabetes <- c("250", seq, list)
t2diabetes_icd9 <- icd9list$coding[which(icd9list$coding %in% diabetes)]


# ischemic heart diseases (I20–I25)
seq <- seq(410,414)
seq2 <- seq(4100,4149)

cardio <- c(seq, seq2)
IHD_icd9 <- icd9list$coding[which(icd9list$coding %in% cardio)]

# ischemic stroke (I63-I64)
seq <- seq(4340,4349)
seq2 <- seq(4360,4369)

ischemic_stroke <- c("434", "436", seq, seq2)
ischemic_stroke_icd9 <- icd9list$coding[which(icd9list$coding %in% ischemic_stroke)]

# Subarachnoid hemorrhage (I60)
seq <- seq(4300,4309)

SH_stroke <- c("430", seq)
SH_stroke_icd9 <- icd9list$coding[which(icd9list$coding %in% SH_stroke)]

# Intracerebral hemorrhage (I61)
seq <- seq(4310,4319)

IH_stroke <- c("431", seq)
IH_stroke_icd9 <- icd9list$coding[which(icd9list$coding %in% IH_stroke)]

# all stroke
all_stroke_icd9 <- icd9list$coding[which(icd9list$coding %in% c(ischemic_stroke, SH_stroke, IH_stroke))]


# COPD and emphysema (J43–J44)
prefix <- "492"
suffix <- sprintf('%0.1d', 0:9)
seq <- paste0(prefix, suffix)

prefix <- "496"
suffix <- sprintf('%0.1d', 0:9)
seq2 <- paste0(prefix, suffix)

emphysema_COPD <- c("492", "496", seq, seq2)
emphysema_COPD_icd9 <- icd9list$coding[which(icd9list$coding %in% emphysema_COPD)]

# COPD (J44)
prefix <- "496"
suffix <- sprintf('%0.1d', 0:9)
seq <- paste0(prefix, suffix)

COPD <- c("496", seq)
COPD_icd9 <- icd9list$coding[which(icd9list$coding %in% COPD)]

# Emphysema (J43)
prefix <- "492"
suffix <- sprintf('%0.1d', 0:9)
seq <- paste0(prefix, suffix)

emphysema <- c("492", seq)
emphysema_icd9 <- icd9list$coding[which(icd9list$coding %in% emphysema)]


# chronic liver disease (K70, K73-K74)
prefix <- "571"
suffix <- sprintf('%0.1d', 0:9)
seq <- paste0(prefix, suffix)

liver <- c("571", seq)
liver_icd9 <- icd9list$coding[which(icd9list$coding %in% liver)]


# chronic kidney disease (N18)
prefix <- "585"
suffix <- sprintf('%0.1d', 0:9)
seq <- paste0(prefix, suffix)

kidney <- c("585", seq)
kidney_icd9 <- icd9list$coding[which(icd9list$coding %in% kidney)]


# all-cause dementia
all_dementia_icd9 <- 
  c("2902",
    "2903",
    "2904",
    "2912",
    "2941",
    "3310",
    "3311",
    "3312",
    "3315")

# vascular dementia 
vasc_dementia_icd9 <- c("2904")

# Alzheimer's
alzheimers_icd9 <- c("3310")

# Parkinsons (G20-G22)
prefix <- "332"
suffix <- sprintf('%0.1d', 0:9)
seq <- paste0(prefix, suffix)

parkinsons <- c("332", seq)
parkinsons_icd9 <- icd9list$coding[which(icd9list$coding %in% parkinsons)]


# rheumatoid arthritis (M05-06)
prefix <- "714"
suffix <- sprintf('%0.1d', 0:9)
seq <- paste0(prefix, suffix)

list <- as.list(rep(1, length(seq)))
for (k in seq_along(seq)) {
  list[[k]] <- paste0(seq[k], sprintf('%0.1d', 0:9))
}

list <- unlist(list)

rheumatoid <- c("714",
                seq,
                list)
rheumatoid_icd9 <- icd9list$coding[which(icd9list$coding %in% rheumatoid)]


# macular degeneration (H35.3)
macular <- "3625"
macular_degen_icd9 <- icd9list$coding[which(icd9list$coding %in% macular)]


# osteoporosis (M80-M81)
prefix <- "7330"
suffix <- sprintf('%0.1d', 0:9)
seq <- paste0(prefix, suffix)

osteoporosis <- c("7330",
                  seq)
osteoporosis_icd9 <- icd9list$coding[which(icd9list$coding %in% osteoporosis)]


# osteoarthritis (M15-M19)
prefix <- "715"
suffix <- sprintf('%0.1d', 0:9)
seq <- paste0(prefix, suffix)

list <- as.list(rep(1, length(seq)))
for (k in seq_along(seq)) {
  list[[k]] <- paste0(seq[k], sprintf('%0.1d', 0:9))
}

list <- unlist(list)

osteoarthritis <- c("715",
                    seq,
                    list)
osteoarthritis_icd9 <- icd9list$coding[which(icd9list$coding %in% osteoarthritis)]


## clinical endophenotypes

# hypertension (I10–I15)
seq <- seq(401,405)
seq2 <- seq(4010,4059)

hypertension <- c(seq, seq2)
hypertension_icd9 <- icd9list$coding[which(icd9list$coding %in% hypertension)]

# obesity (E66)
obesity_icd9 <- "2780"

# dyslipidemia (E78)
prefix <- "272"
suffix <- sprintf('%0.1d', 0:9)
seq <- paste0(prefix, suffix)

list <- as.list(rep(1, length(seq)))
for (k in seq_along(seq)) {
  list[[k]] <- paste0(seq[k], sprintf('%0.1d', 0:9))
}

list <- unlist(list)

dyslipidemia <- c("272",
                  seq,
                  list)

dyslipidemia_icd9 <- icd9list$coding[which(icd9list$coding %in% dyslipidemia)]


# create list of all NCD codes
icd9_include <- list(
  IBD_icd9,
  endometriosis_icd9,
  t2diabetes_icd9,
  IHD_icd9,
  ischemic_stroke_icd9,
  IH_stroke_icd9,
  SH_stroke_icd9,
  all_stroke_icd9,
  emphysema_COPD_icd9,
  COPD_icd9,
  emphysema_icd9,
  liver_icd9, 
  kidney_icd9,
  all_dementia_icd9,
  vasc_dementia_icd9,
  alzheimers_icd9,
  parkinsons_icd9,
  rheumatoid_icd9,
  macular_degen_icd9,
  osteoporosis_icd9,
  osteoarthritis_icd9,
  hypertension_icd9,
  obesity_icd9,
  dyslipidemia_icd9
)