# Thesis Analysis Part 1: Data Cleaning and Descriptive Statistics
# Author: Elif Karabacak
# MSc Business Information Management, RSM 2026

setwd("~/Downloads/thesis code 8-5-26")
set.seed(123)

library(dplyr)
library(stringr)
library(ggplot2)
library(readr)

project_dir <- getwd()

investments_file <- file.path(project_dir, "investments_VC .csv")
startups_file <- file.path(project_dir, "Startups .csv")

dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)
dir.create("reports/tables", recursive = TRUE, showWarnings = FALSE)
dir.create("reports/figures", recursive = TRUE, showWarnings = FALSE)


#Helper functions

clean_money <- function(x) {
  x <- as.character(x)
  x[x %in% c("", "-", "NA", "NaN", "nan")] <- NA
  x <- gsub("[^0-9.\\-]", "", x)
  x <- suppressWarnings(as.numeric(x))
  x[is.na(x)] <- 0
  return(x)
}

clean_name <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  x <- tolower(x)
  x <- gsub("&", " and ", x)
  x <- gsub("[^a-z0-9]+", " ", x)
  x <- gsub("\\b(inc|llc|ltd|limited|corp|corporation|company|co)\\b", " ", x)
  x <- gsub("\\s+", " ", x)
  trimws(x)
}

clean_permalink <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  x <- tolower(trimws(x))
  x <- gsub("https?://(www\\.)?crunchbase\\.com", "", x)
  x <- sub("[?#].*$", "", x)
  x <- sub("/+$", "", x)
  x[x != "" & substr(x, 1, 1) != "/"] <- paste0("/", x[x != "" & substr(x, 1, 1) != "/"])
  return(x)
}

get_cb_permalink <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  x <- tolower(x)
  
  ok <- grepl("crunchbase\\.com/(organization|company)/", x)
  slug <- sub(".*crunchbase\\.com/(organization|company)/([^/?#]+).*", "\\2", x)
  
  ifelse(ok, paste0("/organization/", slug), "")
}

get_year <- function(x) {
  y <- suppressWarnings(as.integer(substr(as.character(x), 1, 4)))
  y[y < 1800 | y > 2100] <- NA
  return(y)
}

first_category <- function(category_list, market) {
  out <- ifelse(!is.na(market) & trimws(market) != "", trimws(market), NA)
  
  for (i in which(is.na(out) | out == "")) {
    value <- category_list[i]
    
    if (is.na(value) || trimws(value) == "") {
      out[i] <- "Unknown"
    } else {
      parts <- trimws(unlist(strsplit(value, "\\|")))
      parts <- parts[parts != ""]
      out[i] <- ifelse(length(parts) == 0, "Unknown", parts[1])
    }
  }
  
  out[is.na(out) | out == ""] <- "Unknown"
  return(out)
}

#Load raw data

vc_raw <- read.csv(
  investments_file,
  fileEncoding = "latin1",
  check.names = FALSE,
  stringsAsFactors = FALSE
)

startups_raw <- read.csv(
  startups_file,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

names(vc_raw) <- trimws(names(vc_raw))
names(startups_raw) <- trimws(names(startups_raw))

cat("VC rows:", nrow(vc_raw), "\n")
cat("Startup rows:", nrow(startups_raw), "\n")

#Clean VC investment data

vc <- vc_raw

money_cols <- c(
  "funding_total_usd", "seed", "venture", "angel", "grant",
  "debt_financing", "round_A", "round_B", "round_C", "round_D",
  "round_E", "round_F", "round_G", "round_H"
)

for (col in intersect(money_cols, names(vc))) {
  vc[[col]] <- clean_money(vc[[col]])
}

vc$company_name <- trimws(as.character(vc$name))
vc$name_key <- clean_name(vc$company_name)
vc$permalink_key <- clean_permalink(vc$permalink)

vc$market <- trimws(as.character(vc$market))
vc$primary_category <- first_category(vc$category_list, vc$market)

vc$founded_year <- suppressWarnings(as.numeric(vc$founded_year))
vc$first_funding_year <- get_year(vc$first_funding_at)
vc$last_funding_year <- get_year(vc$last_funding_at)
vc$years_to_first_funding <- vc$first_funding_year - vc$founded_year

vc$seed_amount_usd <- vc$seed
vc$series_a_amount_usd <- vc$round_A

later_rounds <- intersect(
  c("round_B", "round_C", "round_D", "round_E", "round_F", "round_G", "round_H"),
  names(vc)
)

vc$series_b_or_later_amount_usd <- rowSums(vc[, later_rounds, drop = FALSE], na.rm = TRUE)

vc$series_a_success <- ifelse(vc$series_a_amount_usd > 0, 1, 0)

seed_sample <- vc %>%
  filter(seed_amount_usd > 0) %>%
  arrange(permalink_key, desc(funding_total_usd)) %>%
  distinct(permalink_key, .keep_all = TRUE)

structured_seed <- seed_sample %>%
  select(
    permalink, permalink_key, company_name, name_key, homepage_url,
    category_list, market, primary_category, status,
    country_code, state_code, region, city,
    funding_total_usd, funding_rounds,
    founded_year, first_funding_year, last_funding_year,
    years_to_first_funding, seed_amount_usd,
    series_a_amount_usd, series_b_or_later_amount_usd,
    series_a_success
  )

cat("Seed-funded startups:", nrow(structured_seed), "\n")

#Clean startup description data

startups <- startups_raw

startups_clean <- startups %>%
  mutate(
    startup_company = trimws(as.character(Company)),
    name_key = clean_name(startup_company),
    startup_permalink_key = get_cb_permalink(`Crunchbase / Angel List Profile`),
    startup_status = as.character(Satus),
    startup_year_founded = suppressWarnings(as.numeric(`Year Founded`)),
    description = trimws(as.character(Description)),
    startup_categories = as.character(Categories),
    founders = as.character(Founders),
    yc_year = suppressWarnings(as.numeric(`Y Combinator Year`)),
    yc_session = as.character(`Y Combinator Session`),
    investors = as.character(Investors),
    headquarters_city = as.character(`Headquarters (City)`),
    headquarters_state = as.character(`Headquarters (US State)`),
    headquarters_country = as.character(`Headquarters (Country)`),
    startup_website = as.character(Website),
    description_word_count = ifelse(
      is.na(description) | description == "",
      0,
      str_count(description, "\\S+")
    ),
    description_char_count = nchar(description)
  ) %>%
  select(
    startup_company, name_key, startup_permalink_key,
    startup_status, startup_year_founded,
    description, description_word_count, description_char_count,
    startup_categories, founders, yc_year, yc_session,
    investors, headquarters_city, headquarters_state,
    headquarters_country, startup_website
  )


#Match VC data with startup descriptions


startup_by_permalink <- startups_clean %>%
  filter(startup_permalink_key != "") %>%
  distinct(startup_permalink_key, .keep_all = TRUE)

matched_by_permalink <- structured_seed %>%
  inner_join(
    startup_by_permalink,
    by = c("permalink_key" = "startup_permalink_key")
  ) %>%
  mutate(match_method = "crunchbase_permalink")

matched_keys <- unique(matched_by_permalink$permalink_key)

unmatched_seed <- structured_seed %>%
  filter(!(permalink_key %in% matched_keys))

startup_by_name <- startups_clean %>%
  filter(name_key != "") %>%
  distinct(name_key, .keep_all = TRUE)

matched_by_name <- unmatched_seed %>%
  inner_join(startup_by_name, by = "name_key") %>%
  mutate(match_method = "normalized_company_name")

matched_text <- bind_rows(matched_by_permalink, matched_by_name) %>%
  filter(!is.na(description), description != "") %>%
  distinct(permalink_key, .keep_all = TRUE) %>%
  mutate(
    company_id = ifelse(permalink_key != "", permalink_key, name_key)
  )

cat("Matched text sample:", nrow(matched_text), "\n")
cat("Successes:", sum(matched_text$series_a_success == 1), "\n")
cat("Non-successes:", sum(matched_text$series_a_success == 0), "\n")

#Save cleaned files

descriptions_for_embeddings <- matched_text %>%
  select(company_id, company_name, description)

write.csv(
  structured_seed,
  "data/processed/structured_seed_sample_clean.csv",
  row.names = FALSE
)

write.csv(
  matched_text,
  "data/processed/matched_text_sample_clean.csv",
  row.names = FALSE
)

write.csv(
  descriptions_for_embeddings,
  "data/processed/descriptions_for_embeddings.csv",
  row.names = FALSE
)

# Data cleaning summary

summary_table <- data.frame(
  metric = c(
    "raw_vc_rows",
    "raw_startup_description_rows",
    "structured_seed_rows",
    "matched_text_rows",
    "matched_crunchbase_permalink_matches",
    "matched_clean_name_matches",
    "matched_series_a_successes",
    "matched_series_a_non_successes",
    "matched_series_a_success_rate"
  ),
  value = c(
    nrow(vc_raw),
    nrow(startups_raw),
    nrow(structured_seed),
    nrow(matched_text),
    sum(matched_text$match_method == "crunchbase_permalink"),
    sum(matched_text$match_method == "normalized_company_name"),
    sum(matched_text$series_a_success == 1),
    sum(matched_text$series_a_success == 0),
    round(mean(matched_text$series_a_success), 3)
  )
)

print(summary_table)

write.csv(
  summary_table,
  "reports/tables/data_cleaning_summary.csv",
  row.names = FALSE
)

#Descriptive statistics

matched_text$country_code[
  is.na(matched_text$country_code) | matched_text$country_code == ""
] <- "Unknown"

top_countries <- matched_text %>%
  count(country_code, sort = TRUE) %>%
  head(15)

top_categories <- matched_text %>%
  count(primary_category, sort = TRUE) %>%
  head(15)

missing_values <- data.frame(
  variable = names(matched_text),
  missing_count = colSums(is.na(matched_text))
) %>%
  arrange(desc(missing_count))

seed_summary <- summary(matched_text$seed_amount_usd)
description_summary <- summary(matched_text$description_word_count)

print(top_countries)
print(top_categories)
print(seed_summary)
print(description_summary)
print(head(missing_values, 15))

write.csv(top_countries, "reports/tables/top_countries_matched.csv", row.names = FALSE)
write.csv(top_categories, "reports/tables/top_categories_matched.csv", row.names = FALSE)
write.csv(missing_values, "reports/tables/missing_values_matched.csv", row.names = FALSE)

# Figures

ggplot(matched_text, aes(x = factor(series_a_success))) +
  geom_bar(fill = "lightblue") +
  scale_x_discrete(
    labels = c(
      "0" = "No follow-on funding",
      "1" = "Follow-on funding"
    )
  ) +
  labs(
    title = "Distribution of follow-on funding outcomes in the matched sample",
    x = "Funding outcome",
    y = "Number of startups"
  )

ggsave("reports/figures/outcome_balance_matched.png", width = 7, height = 5)

ggplot(matched_text, aes(x = seed_amount_usd / 1000000)) +
  geom_histogram(bins = 30, fill = "lightblue", color = "white") +
  labs(
    title = "Distribution of seed funding",
    x = "Seed funding amount in million USD",
    y = "Number of startups"
  )

ggsave("reports/figures/seed_funding_distribution.png", width = 7, height = 5)

ggplot(matched_text, aes(x = description_word_count)) +
  geom_histogram(bins = 20, fill = "lightblue", color = "white") +
  labs(
    title = "Distribution of startup description length",
    x = "Word count",
    y = "Number of startups"
  )

ggsave("reports/figures/description_length_distribution.png", width = 7, height = 5)

ggplot(top_countries, aes(x = reorder(country_code, n), y = n)) +
  geom_col(fill = "lightblue") +
  coord_flip() +
  labs(
    title = "Top countries in matched sample",
    x = "Country",
    y = "Number of startups"
  )

ggsave("reports/figures/top_countries_matched.png", width = 7, height = 5)

ggplot(top_categories, aes(x = reorder(primary_category, n), y = n)) +
  geom_col(fill = "lightblue") +
  coord_flip() +
  labs(
    title = "Top categories in matched sample",
    x = "Category",
    y = "Number of startups"
  )

ggsave("reports/figures/top_categories_matched.png", width = 8, height = 5)
