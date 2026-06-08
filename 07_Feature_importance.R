#Thesis analysis: Part 7
#Feature importance for text-enhanced XGBoost

library(dplyr)
library(caret)
library(xgboost)
library(ggplot2)

dir.create("reports/models", recursive = TRUE, showWarnings = FALSE)
dir.create("reports/figures", recursive = TRUE, showWarnings = FALSE)

df <- read.csv("data/processed/matched_text_sample_clean.csv")
embeddings <- read.csv("data/processed/sbert_embeddings.csv")

full_df <- df %>%
  inner_join(embeddings, by = "company_id")

structured_numeric_features <- c(
  "founded_year",
  "first_funding_year",
  "years_to_first_funding",
  "seed_amount_usd",
  "funding_rounds"
)

categorical_features <- c(
  "country_code",
  "state_code",
  "region",
  "city",
  "primary_category",
  "market"
)

embedding_cols <- grep("^emb_", names(full_df), value = TRUE)

full_df$series_a_success <- factor(
  ifelse(full_df$series_a_success == 1, "Yes", "No"),
  levels = c("No", "Yes")
)

y <- full_df$series_a_success
y_num <- ifelse(y == "Yes", 1, 0)

#Structured numeric features

x_num <- full_df[, structured_numeric_features]

for (col in structured_numeric_features) {
  x_num[[col]] <- as.numeric(x_num[[col]])
  med <- median(x_num[[col]], na.rm = TRUE)
  if (is.na(med)) med <- 0
  x_num[[col]][is.na(x_num[[col]])] <- med
}

#Structured categorical features

x_cat <- full_df[, categorical_features]

for (col in categorical_features) {
  x_cat[[col]] <- as.character(x_cat[[col]])
  x_cat[[col]][is.na(x_cat[[col]]) | x_cat[[col]] == ""] <- "Unknown"
  x_cat[[col]] <- as.factor(x_cat[[col]])
}

dummy_encoder <- dummyVars(~ ., data = x_cat)

x_cat_encoded <- predict(dummy_encoder, newdata = x_cat)
x_cat_encoded <- as.data.frame(x_cat_encoded)
names(x_cat_encoded) <- make.names(names(x_cat_encoded), unique = TRUE)

# PCA on embeddings for final interpretability model

emb_matrix <- as.matrix(full_df[, embedding_cols])

pca_model <- prcomp(
  emb_matrix,
  center = TRUE,
  scale. = TRUE
)

train_pca <- pca_model$x[, 1:30, drop = FALSE]
colnames(train_pca) <- paste0("text_pc_", seq_len(30))

# Combine features

x <- cbind(
  x_num,
  x_cat_encoded,
  train_pca
)

x <- as.data.frame(x)
names(x) <- make.names(names(x), unique = TRUE)

zero_var <- nearZeroVar(x)

if (length(zero_var) > 0) {
  x <- x[, -zero_var, drop = FALSE]
}

for (col in names(x)) {
  x[[col]] <- as.numeric(x[[col]])
  med <- median(x[[col]], na.rm = TRUE)
  if (is.na(med)) med <- 0
  x[[col]][is.na(x[[col]])] <- med
}

scale_pos_weight <- sum(y_num == 0) / sum(y_num == 1)

# Train final XGBoost model

xgb_final <- xgboost(
  x = as.matrix(x),
  y = factor(y_num, levels = c(0, 1)),
  objective = "binary:logistic",
  eval_metric = "auc",
  nrounds = 100,
  max_depth = 3,
  learning_rate = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8,
  scale_pos_weight = scale_pos_weight,
  verbosity = 0
)

importance <- xgb.importance(
  feature_names = names(x),
  model = xgb_final
)

importance$feature_type <- ifelse(
  grepl("^text_pc_", importance$Feature),
  "Text embedding",
  "Structured"
)

print(head(importance, 20))

write.csv(
  importance,
  "reports/models/xgboost_text_feature_importance.csv",
  row.names = FALSE
)

top_importance <- importance %>%
  head(20)

ggplot(
  top_importance,
  aes(x = reorder(Feature, Gain), y = Gain, fill = feature_type)
) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(
    values = c(
      "Structured" = "lightblue",
      "Text embedding" = "darkblue"
    )
  ) +
  labs(
    title = "Top 20 features in text-enhanced XGBoost model",
    x = "Feature",
    y = "Importance gain",
    fill = "Feature type"
  )

ggsave(
  "reports/figures/xgboost_text_feature_importance.png",
  width = 8,
  height = 6
)

feature_type_summary <- importance %>%
  group_by(feature_type) %>%
  summarise(
    total_gain = sum(Gain),
    .groups = "drop"
  ) %>%
  mutate(
    gain_share = total_gain / sum(total_gain)
  )

print(feature_type_summary)

write.csv(
  feature_type_summary,
  "reports/models/xgboost_feature_type_summary.csv",
  row.names = FALSE
)

# SHAP analysis using XGBoost contribution values
x_shap <- as.matrix(x)
colnames(x_shap) <- names(x)

shap_contrib <- predict(
  xgb_final,
  newdata = x_shap,
  type = "contrib"
)

#Convert to data frame
shap_contrib <- as.data.frame(shap_contrib)

#Remove intercept column 
if (ncol(shap_contrib) == ncol(x_shap) + 1) {
  shap_values_only <- shap_contrib[, -ncol(shap_contrib), drop = FALSE]
} else if (ncol(shap_contrib) == ncol(x_shap)) {
  shap_values_only <- shap_contrib
} else {
  stop("SHAP output dimensions do not match feature matrix. Check xgboost predict output.")
}

#Assign feature names
colnames(shap_values_only) <- colnames(x_shap)

#Save raw SHAP values
write.csv(
  shap_values_only,
  "reports/models/xgboost_shap_values.csv",
  row.names = FALSE
)

# Mean absolute SHAP importance
shap_importance <- data.frame(
  Feature = colnames(shap_values_only),
  mean_abs_shap = colMeans(abs(shap_values_only))
) %>%
  arrange(desc(mean_abs_shap)) %>%
  mutate(
    feature_type = ifelse(
      grepl("^text_pc_", Feature),
      "Text embedding",
      "Structured"
    )
  )

print(head(shap_importance, 20))

write.csv(
  shap_importance,
  "reports/models/xgboost_shap_importance.csv",
  row.names = FALSE
)

#Top 20 SHAP feature importance plot

top_shap <- shap_importance %>%
  head(20)

ggplot(
  top_shap,
  aes(
    x = reorder(Feature, mean_abs_shap),
    y = mean_abs_shap,
    fill = feature_type
  )
) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(
    values = c(
      "Structured" = "lightblue",
      "Text embedding" = "darkblue"
    )
  ) +
  labs(
    title = "Top 20 features in text-enhanced XGBoost model based on SHAP values",
    x = "Feature",
    y = "Mean absolute SHAP value",
    fill = "Feature type"
  )

ggsave(
  "reports/figures/xgboost_shap_importance.png",
  width = 8,
  height = 6
)

#SHAP importance by feature type
shap_feature_type_summary <- shap_importance %>%
  group_by(feature_type) %>%
  summarise(
    total_mean_abs_shap = sum(mean_abs_shap),
    .groups = "drop"
  ) %>%
  mutate(
    shap_share = total_mean_abs_shap / sum(total_mean_abs_shap)
  )

print(shap_feature_type_summary)

write.csv(
  shap_feature_type_summary,
  "reports/models/xgboost_shap_feature_type_summary.csv",
  row.names = FALSE
)
