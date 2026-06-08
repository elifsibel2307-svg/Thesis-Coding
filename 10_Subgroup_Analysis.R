#Thesis analysis - Part 9
#Exploratory subgroup analysis for H2 / RQ5
# Subgroups:
#Seed funding size: lower vs higher seed funding
#Structured data completeness: complete vs sparse profile
set.seed(123)

library(dplyr)
library(caret)
library(xgboost)
library(pROC)

dir.create("reports/models", recursive = TRUE, showWarnings = FALSE)

#Load data
df <- read.csv("data/processed/matched_text_sample_clean.csv")
embeddings <- read.csv("data/processed/sbert_embeddings.csv")

full_df <- df %>%
  inner_join(embeddings, by = "company_id")

#Outcome variable

full_df$series_a_success <- factor(
  ifelse(full_df$series_a_success == 1, "Yes", "No"),
  levels = c("No", "Yes")
)

y <- full_df$series_a_success

#Subgroups
#Subgroup A: seed funding size
seed_median <- median(full_df$seed_amount_usd, na.rm = TRUE)

full_df$subgroup_size <- ifelse(
  full_df$seed_amount_usd <= seed_median,
  "Lower seed funding",
  "Higher seed funding"
)

full_df$subgroup_size <- factor(
  full_df$subgroup_size,
  levels = c("Lower seed funding", "Higher seed funding")
)

#Subgroup B: structured data completeness
full_df$has_country <- !is.na(full_df$country_code) &
  trimws(full_df$country_code) != "" &
  trimws(full_df$country_code) != "Unknown"

full_df$has_category <- !is.na(full_df$primary_category) &
  trimws(full_df$primary_category) != "" &
  trimws(full_df$primary_category) != "Unknown"

full_df$subgroup_completeness <- ifelse(
  full_df$has_country & full_df$has_category,
  "Complete structured profile",
  "Sparse structured profile"
)

full_df$subgroup_completeness <- factor(
  full_df$subgroup_completeness,
  levels = c("Complete structured profile", "Sparse structured profile")
)

cat("\n--- Subgroup A: Seed funding size ---\n")
print(table(full_df$subgroup_size))
print(table(full_df$subgroup_size, full_df$series_a_success))

cat("\n--- Subgroup B: Data completeness ---\n")
print(table(full_df$subgroup_completeness))
print(table(full_df$subgroup_completeness, full_df$series_a_success))

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

calculate_metrics <- function(actual, prob, threshold = 0.5) {
  
  prob <- as.numeric(prob)
  prob[is.na(prob)] <- mean(actual == "Yes")
  
  pred <- factor(
    ifelse(prob >= threshold, "Yes", "No"),
    levels = c("No", "Yes")
  )
  
  actual <- factor(actual, levels = c("No", "Yes"))
  
  cm <- confusionMatrix(pred, actual, positive = "Yes")
  
  auc <- if (length(unique(actual)) == 2) {
    as.numeric(
      roc(actual, prob, levels = c("No", "Yes"), quiet = TRUE)$auc
    )
  } else {
    NA
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
  
  #Numeric structured features
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
  
  #Categorical structured features
  x_train_cat <- train_df[, categorical_features]
  x_test_cat <- test_df[, categorical_features]
  
  for (col in categorical_features) {
    x_train_cat[[col]] <- as.character(x_train_cat[[col]])
    x_test_cat[[col]] <- as.character(x_test_cat[[col]])
    
    x_train_cat[[col]][
      is.na(x_train_cat[[col]]) | x_train_cat[[col]] == ""
    ] <- "Unknown"
    
    x_test_cat[[col]][
      is.na(x_test_cat[[col]]) | x_test_cat[[col]] == ""
    ] <- "Unknown"
    
    x_test_cat[[col]][
      !(x_test_cat[[col]] %in% x_train_cat[[col]])
    ] <- "Unknown"
    
    train_levels <- unique(c(x_train_cat[[col]], "Unknown"))
    
    x_train_cat[[col]] <- factor(
      x_train_cat[[col]],
      levels = train_levels
    )
    
    x_test_cat[[col]] <- factor(
      x_test_cat[[col]],
      levels = train_levels
    )
  }
  
  dummy_encoder <- dummyVars(~ ., data = x_train_cat)
  
  x_train_cat_encoded <- as.data.frame(
    predict(dummy_encoder, newdata = x_train_cat)
  )
  
  x_test_cat_encoded <- as.data.frame(
    predict(dummy_encoder, newdata = x_test_cat)
  )
  
  names(x_train_cat_encoded) <- make.names(
    names(x_train_cat_encoded),
    unique = TRUE
  )
  
  names(x_test_cat_encoded) <- make.names(
    names(x_test_cat_encoded),
    unique = TRUE
  )
  
  x_test_cat_encoded <- x_test_cat_encoded[
    ,
    names(x_train_cat_encoded),
    drop = FALSE
  ]
  
  x_train <- cbind(
    x_train_num,
    x_train_cat_encoded
  )
  
  x_test <- cbind(
    x_test_num,
    x_test_cat_encoded
  )
  
  n_components <- 0
  
  if (include_text) {
    
    emb_train <- as.matrix(train_df[, embedding_cols])
    emb_test <- as.matrix(test_df[, embedding_cols])
    
    pca_model <- prcomp(
      emb_train,
      center = TRUE,
      scale. = TRUE
    )
    
    var_exp <- cumsum(
      pca_model$sdev^2 / sum(pca_model$sdev^2)
    )
    
    n_components <- which(var_exp >= 0.95)[1]
    n_components <- min(n_components, 30)
    
    train_pca <- predict(
      pca_model,
      emb_train
    )[, 1:n_components, drop = FALSE]
    
    test_pca <- predict(
      pca_model,
      emb_test
    )[, 1:n_components, drop = FALSE]
    
    colnames(train_pca) <- paste0(
      "text_pc_",
      seq_len(n_components)
    )
    
    colnames(test_pca) <- paste0(
      "text_pc_",
      seq_len(n_components)
    )
    
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
  
  y_num <- ifelse(y_train == "Yes", 1, 0)
  
  scale_pos_weight <- sum(y_num == 0) /
    sum(y_num == 1)
  
  model <- xgboost(
    x = as.matrix(x_train),
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
  
  as.numeric(
    predict(model, newdata = as.matrix(x_test))
  )
}

#Run CV for a subgroup variable

run_subgroup_cv <- function(full_df, y, subgroup_col, subgroup_label) {
  
  folds <- createFolds(y, k = 5, returnTrain = FALSE)
  
  all_predictions <- list()
  pred_counter <- 1
  
  for (fold_id in seq_along(folds)) {
    
    test_idx <- folds[[fold_id]]
    train_idx <- setdiff(seq_len(nrow(full_df)), test_idx)
    
    train_df <- full_df[train_idx, ]
    test_df <- full_df[test_idx, ]
    
    y_train <- y[train_idx]
    y_test <- y[test_idx]
    
    for (use_text in c(FALSE, TRUE)) {
      
      mat <- build_fold_matrix(
        train_df,
        test_df,
        include_text = use_text
      )
      
      probs <- run_xgboost_fold(
        mat$x_train,
        mat$x_test,
        y_train
      )
      
      all_predictions[[pred_counter]] <- data.frame(
        company_id = test_df$company_id,
        fold = fold_id,
        model = "XGBoost",
        feature_set = ifelse(
          use_text,
          "Structured + SBERT",
          "Structured only"
        ),
        subgroup = test_df[[subgroup_col]],
        actual = y_test,
        predicted_probability = probs,
        predicted_class = ifelse(
          probs >= 0.5,
          "Yes",
          "No"
        )
      )
      
      pred_counter <- pred_counter + 1
    }
  }
  
  predictions <- bind_rows(all_predictions)
  
  results <- predictions %>%
    group_by(subgroup, feature_set, model) %>%
    group_modify(
      ~ calculate_metrics(
        .x$actual,
        .x$predicted_probability
      )
    ) %>%
    ungroup()
  
  base_cols <- c(
    "subgroup",
    "roc_auc",
    "accuracy",
    "precision",
    "recall",
    "f1"
  )
  
  differences <- results %>%
    filter(feature_set == "Structured + SBERT") %>%
    select(all_of(base_cols)) %>%
    inner_join(
      results %>%
        filter(feature_set == "Structured only") %>%
        select(all_of(base_cols)),
      by = "subgroup",
      suffix = c("_text", "_structured")
    ) %>%
    mutate(
      roc_auc_diff = roc_auc_text - roc_auc_structured,
      accuracy_diff = accuracy_text - accuracy_structured,
      precision_diff = precision_text - precision_structured,
      recall_diff = recall_text - recall_structured,
      f1_diff = f1_text - f1_structured
    ) %>%
    select(
      subgroup,
      roc_auc_diff,
      accuracy_diff,
      precision_diff,
      recall_diff,
      f1_diff
    )
  
  cat("\nResults by subgroup:\n")
  print(
    results %>%
      select(
        subgroup,
        feature_set,
        roc_auc,
        accuracy,
        precision,
        recall,
        f1
      )
  )
  
  cat("\nDifferences text-enhanced minus structured-only:\n")
  print(differences)
  
  list(
    predictions = predictions,
    results = results,
    differences = differences
  )
}

#Run both subgroup analyses

results_size <- run_subgroup_cv(
  full_df = full_df,
  y = y,
  subgroup_col = "subgroup_size",
  subgroup_label = "Seed funding size"
)

results_completeness <- run_subgroup_cv(
  full_df = full_df,
  y = y,
  subgroup_col = "subgroup_completeness",
  subgroup_label = "Structured data completeness"
)

#Save outputs

write.csv(
  results_size$predictions,
  "reports/models/subgroup_size_predictions.csv",
  row.names = FALSE
)

write.csv(
  results_size$results,
  "reports/models/subgroup_size_results.csv",
  row.names = FALSE
)

write.csv(
  results_size$differences,
  "reports/models/subgroup_size_differences.csv",
  row.names = FALSE
)

write.csv(
  results_completeness$predictions,
  "reports/models/subgroup_completeness_predictions.csv",
  row.names = FALSE
)

write.csv(
  results_completeness$results,
  "reports/models/subgroup_completeness_results.csv",
  row.names = FALSE
)

write.csv(
  results_completeness$differences,
  "reports/models/subgroup_completeness_differences.csv",
  row.names = FALSE
)

#Summary table 

summary_size <- results_size$differences %>%
  mutate(split_type = "Seed funding size")

summary_completeness <- results_completeness$differences %>%
  mutate(split_type = "Structured data completeness")

combined_summary <- bind_rows(
  summary_size,
  summary_completeness
) %>%
  select(
    split_type,
    subgroup,
    roc_auc_diff,
    accuracy_diff,
    precision_diff,
    recall_diff,
    f1_diff
  ) %>%
  mutate(
    across(
      where(is.numeric),
      ~ round(.x, 4)
    )
  )

print(combined_summary)

write.csv(
  combined_summary,
  "reports/models/subgroup_combined_summary.csv",
  row.names = FALSE
)
