#Thesis analysis: Part 9
#Exploratory subgroup analysis by seed funding size

set.seed(123)

library(dplyr)
library(caret)
library(pROC)
library(xgboost)

dir.create("reports/models", recursive = TRUE, showWarnings = FALSE)

#Load data

df <- read.csv("data/processed/matched_text_sample_clean.csv")
embeddings <- read.csv("data/processed/sbert_embeddings.csv")

full_df <- df %>%
  inner_join(embeddings, by = "company_id")

#Define outcome and subgroup

full_df$series_a_success <- factor(
  ifelse(full_df$series_a_success == 1, "Yes", "No"),
  levels = c("No", "Yes")
)

y <- full_df$series_a_success

seed_median <- median(full_df$seed_amount_usd, na.rm = TRUE)

full_df$seed_funding_group <- ifelse(
  full_df$seed_amount_usd <= seed_median,
  "Lower seed funding",
  "Higher seed funding"
)

full_df$seed_funding_group <- factor(
  full_df$seed_funding_group,
  levels = c("Lower seed funding", "Higher seed funding")
)

cat("Median seed funding:", seed_median, "\n")
print(table(full_df$seed_funding_group))
print(table(full_df$seed_funding_group, full_df$series_a_success))

#Feature definitions

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

#Helper functions
calculate_metrics <- function(actual, prob, threshold = 0.5) {
  
  prob <- as.numeric(prob)
  prob[is.na(prob)] <- mean(actual == "Yes")
  
  pred <- ifelse(prob >= threshold, "Yes", "No")
  pred <- factor(pred, levels = c("No", "Yes"))
  actual <- factor(actual, levels = c("No", "Yes"))
  
  cm <- confusionMatrix(pred, actual, positive = "Yes")
  
  auc <- NA
  
  if (length(unique(actual)) == 2) {
    auc <- as.numeric(
      roc(actual, prob, levels = c("No", "Yes"), quiet = TRUE)$auc
    )
  }
  
  data.frame(
    roc_auc = auc,
    accuracy = as.numeric(cm$overall["Accuracy"]),
    precision = as.numeric(cm$byClass["Precision"]),
    recall = as.numeric(cm$byClass["Recall"]),
    f1 = as.numeric(cm$byClass["F1"]),
    tp = as.numeric(cm$table["Yes", "Yes"]),
    tn = as.numeric(cm$table["No", "No"]),
    fp = as.numeric(cm$table["Yes", "No"]),
    fn = as.numeric(cm$table["No", "Yes"])
  )
}

build_fold_matrix <- function(train_df, test_df, include_text = FALSE) {
  
  #Numeric features
  x_train_num <- train_df[, structured_numeric_features]
  x_test_num <- test_df[, structured_numeric_features]
  
  for (col in structured_numeric_features) {
    x_train_num[[col]] <- as.numeric(x_train_num[[col]])
    x_test_num[[col]] <- as.numeric(x_test_num[[col]])
    
    med <- median(x_train_num[[col]], na.rm = TRUE)
    if (is.na(med)) med <- 0
    
    x_train_num[[col]][is.na(x_train_num[[col]])] <- med
    x_test_num[[col]][is.na(x_test_num[[col]])] <- med
  }
  
  #Categorical features
  x_train_cat <- train_df[, categorical_features]
  x_test_cat <- test_df[, categorical_features]
  
  for (col in categorical_features) {
    x_train_cat[[col]] <- as.character(x_train_cat[[col]])
    x_test_cat[[col]] <- as.character(x_test_cat[[col]])
    
    x_train_cat[[col]][is.na(x_train_cat[[col]]) | x_train_cat[[col]] == ""] <- "Unknown"
    x_test_cat[[col]][is.na(x_test_cat[[col]]) | x_test_cat[[col]] == ""] <- "Unknown"
    
    x_test_cat[[col]][!(x_test_cat[[col]] %in% x_train_cat[[col]])] <- "Unknown"
    
    train_levels <- unique(c(x_train_cat[[col]], "Unknown"))
    
    x_train_cat[[col]] <- factor(x_train_cat[[col]], levels = train_levels)
    x_test_cat[[col]] <- factor(x_test_cat[[col]], levels = train_levels)
  }
  
  dummy_encoder <- dummyVars(~ ., data = x_train_cat)
  
  x_train_cat_encoded <- as.data.frame(
    predict(dummy_encoder, newdata = x_train_cat)
  )
  
  x_test_cat_encoded <- as.data.frame(
    predict(dummy_encoder, newdata = x_test_cat)
  )
  
  names(x_train_cat_encoded) <- make.names(names(x_train_cat_encoded), unique = TRUE)
  names(x_test_cat_encoded) <- make.names(names(x_test_cat_encoded), unique = TRUE)
  
  x_test_cat_encoded <- x_test_cat_encoded[, names(x_train_cat_encoded), drop = FALSE]
  
  x_train <- cbind(
    x_train_num,
    x_train_cat_encoded
  )
  
  x_test <- cbind(
    x_test_num,
    x_test_cat_encoded
  )
  
  #Add SBERT PCA features 
  n_components <- 0
  
  if (include_text) {
    
    emb_train <- as.matrix(train_df[, embedding_cols])
    emb_test <- as.matrix(test_df[, embedding_cols])
    
    pca_model <- prcomp(
      emb_train,
      center = TRUE,
      scale. = TRUE
    )
    
    explained_variance <- cumsum(
      pca_model$sdev^2 / sum(pca_model$sdev^2)
    )
    
    n_components <- which(explained_variance >= 0.95)[1]
    n_components <- min(n_components, 30)
    
    train_pca <- predict(
      pca_model,
      emb_train
    )[, 1:n_components, drop = FALSE]
    
    test_pca <- predict(
      pca_model,
      emb_test
    )[, 1:n_components, drop = FALSE]
    
    colnames(train_pca) <- paste0("text_pc_", seq_len(n_components))
    colnames(test_pca) <- paste0("text_pc_", seq_len(n_components))
    
    x_train <- cbind(x_train, train_pca)
    x_test <- cbind(x_test, test_pca)
  }
  
  x_train <- as.data.frame(x_train)
  x_test <- as.data.frame(x_test)
  
  names(x_train) <- make.names(names(x_train), unique = TRUE)
  names(x_test) <- make.names(names(x_test), unique = TRUE)
  
  x_test <- x_test[, names(x_train), drop = FALSE]
  
  zero_var <- nearZeroVar(x_train)
  
  if (length(zero_var) > 0) {
    x_train <- x_train[, -zero_var, drop = FALSE]
    x_test <- x_test[, names(x_train), drop = FALSE]
  }
  
  #Safety check
  for (col in names(x_train)) {
    x_train[[col]] <- as.numeric(x_train[[col]])
    x_test[[col]] <- as.numeric(x_test[[col]])
    
    med <- median(x_train[[col]], na.rm = TRUE)
    if (is.na(med)) med <- 0
    
    x_train[[col]][is.na(x_train[[col]])] <- med
    x_test[[col]][is.na(x_test[[col]])] <- med
  }
  
  list(
    x_train = x_train,
    x_test = x_test,
    n_components = n_components
  )
}

run_xgboost_fold <- function(x_train, x_test, y_train) {
  
  y_train_num <- ifelse(y_train == "Yes", 1, 0)
  
  scale_pos_weight <- sum(y_train_num == 0) /
    sum(y_train_num == 1)
  
  model <- xgboost(
    x = as.matrix(x_train),
    y = factor(y_train_num, levels = c(0, 1)),
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
  
  as.numeric(
    predict(model, newdata = as.matrix(x_test))
  )
}

#Cross-validation

folds <- createFolds(y, k = 5, returnTrain = FALSE)

all_predictions <- list()
pred_counter <- 1

for (fold_id in seq_along(folds)) {
  
  cat("Running fold", fold_id, "\n")
  
  test_idx <- folds[[fold_id]]
  train_idx <- setdiff(seq_len(nrow(full_df)), test_idx)
  
  train_df <- full_df[train_idx, ]
  test_df <- full_df[test_idx, ]
  
  y_train <- y[train_idx]
  y_test <- y[test_idx]
  
  #Structured-only XGBoost
  
  structured_matrix <- build_fold_matrix(
    train_df,
    test_df,
    include_text = FALSE
  )
  
  structured_prob <- run_xgboost_fold(
    structured_matrix$x_train,
    structured_matrix$x_test,
    y_train
  )
  
  all_predictions[[pred_counter]] <- data.frame(
    company_id = test_df$company_id,
    fold = fold_id,
    model = "XGBoost",
    feature_set = "Structured only",
    subgroup = test_df$seed_funding_group,
    actual = y_test,
    predicted_probability = structured_prob,
    predicted_class = ifelse(structured_prob >= 0.5, "Yes", "No")
  )
  
  pred_counter <- pred_counter + 1
  
  #Text-enhanced XGBoost
  
  text_matrix <- build_fold_matrix(
    train_df,
    test_df,
    include_text = TRUE
  )
  
  text_prob <- run_xgboost_fold(
    text_matrix$x_train,
    text_matrix$x_test,
    y_train
  )
  
  all_predictions[[pred_counter]] <- data.frame(
    company_id = test_df$company_id,
    fold = fold_id,
    model = "XGBoost",
    feature_set = "Structured + SBERT",
    subgroup = test_df$seed_funding_group,
    actual = y_test,
    predicted_probability = text_prob,
    predicted_class = ifelse(text_prob >= 0.5, "Yes", "No"),
    n_text_pcs = text_matrix$n_components
  )
  
  pred_counter <- pred_counter + 1
}

subgroup_predictions <- bind_rows(all_predictions)

#Calculate subgroup metrics

subgroup_results <- subgroup_predictions %>%
  group_by(subgroup, feature_set, model) %>%
  group_modify(
    ~ calculate_metrics(
      actual = .x$actual,
      prob = .x$predicted_probability
    )
  ) %>%
  ungroup()

print(subgroup_results)

write.csv(
  subgroup_predictions,
  "reports/models/subgroup_seed_funding_predictions.csv",
  row.names = FALSE
)

write.csv(
  subgroup_results,
  "reports/models/subgroup_seed_funding_results.csv",
  row.names = FALSE
)

#Compare text-enhanced minus structured-only

structured_subgroup <- subgroup_results %>%
  filter(feature_set == "Structured only") %>%
  select(
    subgroup,
    roc_auc,
    accuracy,
    precision,
    recall,
    f1
  )

text_subgroup <- subgroup_results %>%
  filter(feature_set == "Structured + SBERT") %>%
  select(
    subgroup,
    roc_auc,
    accuracy,
    precision,
    recall,
    f1
  )

subgroup_differences <- text_subgroup %>%
  inner_join(
    structured_subgroup,
    by = "subgroup",
    suffix = c("_text", "_structured")
  ) %>%
  mutate(
    roc_auc_difference = roc_auc_text - roc_auc_structured,
    accuracy_difference = accuracy_text - accuracy_structured,
    precision_difference = precision_text - precision_structured,
    recall_difference = recall_text - recall_structured,
    f1_difference = f1_text - f1_structured
  ) %>%
  select(
    subgroup,
    roc_auc_difference,
    accuracy_difference,
    precision_difference,
    recall_difference,
    f1_difference
  )

print(subgroup_differences)

write.csv(
  subgroup_differences,
  "reports/models/subgroup_seed_funding_differences.csv",
  row.names = FALSE
)