library(lubridate)
library(data.table)

# specify columns to load
cols <- c(
  'eid',
  'p34',
  'p52',
  'p53_i0',
  'p53_i2',
  'p53_i3',
  'p21022'
)

# load raw data
raw_data <- fread(
  ".../UKB/Datasets/Data tables/ukb_61054_raw_phenotype_data_june_14_2023.csv",
  select = cols)

raw_data$birth_year <- raw_data$p34
raw_data$birth_month <- raw_data$p52
raw_data$recruitment_date <- raw_data$p53_i0
raw_data$recruitment_date_ins2 <- raw_data$p53_i2
raw_data$recruitment_date_ins3 <- raw_data$p53_i3
raw_data$recruitment_age <- raw_data$p21022

# recode as date
head(raw_data$recruitment_date)
raw_data$recruitment_date <- as.Date(raw_data$recruitment_date, format = "%Y-%m-%d")
head(raw_data$recruitment_date_ins2)
raw_data$recruitment_date_ins2 <- as.Date(raw_data$recruitment_date_ins2, format = "%Y-%m-%d")
head(raw_data$recruitment_date_ins3)
raw_data$recruitment_date_ins3 <- as.Date(raw_data$recruitment_date_ins3, format = "%Y-%m-%d")

# make estimated birth date a first day of birth month and year
raw_data$est_birth_date <- ymd(paste0(raw_data$birth_year, "-", raw_data$birth_month, "-01"))
head(raw_data$est_birth_date)
raw_data$est_birth_date <- as.Date(raw_data$est_birth_date, format = "%Y-%m-%d")

# make granular age as difference between recruitment date and estimated birth date
raw_data$age_granular <- as.numeric(raw_data$recruitment_date - raw_data$est_birth_date)
raw_data$age_granular <- raw_data$age_granular/365.25

# code age at 2014+ imaging visit (ins = 2)
raw_data$time2 <- as.numeric(raw_data$recruitment_date_ins2 - raw_data$recruitment_date) / 365.25
raw_data$age_granular_ins2 <- raw_data$age_granular + raw_data$time2

# code age at 2019+ repeat imaging visit (ins = 3)
raw_data$time3 <- as.numeric(raw_data$recruitment_date_ins3 - raw_data$recruitment_date) / 365.25
raw_data$age_granular_ins3 <- raw_data$age_granular + raw_data$time3

cols <- c(
  "eid",
  "birth_year",
  "birth_month",
  "recruitment_date",
  "recruitment_date_ins2",
  "recruitment_date_ins3",
  "recruitment_age",
  "est_birth_date",
  "age_granular",
  "age_granular_ins2",
  "age_granular_ins3"
)

# subset to cols of interest
data <- as.data.frame(raw_data)[cols]

# remove 1 person with NA for age and birth month
data <- data[which(!is.na(data$recruitment_age)), ]

# check distributions
hist(data$recruitment_age)
hist(data$age_granular)

summary(data$recruitment_age)
summary(data$age_granular)

# save
write.csv(data, row.names = FALSE, ".../UKB/Datasets/6. Outcome data/granular_age_july_23_2023.csv")

