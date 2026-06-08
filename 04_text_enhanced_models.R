# Thesis analysis - Part 4
# Text-enhanced models using SBERT + PCA

set.seed(123)

library(dplyr)
library(caret)
library(pROC)
library(randomForest)
library(xgboost)

dir.create("reports/models", recursive = TRUE, showWarnings = FALSE)

#Load data

df <- read.csv("data/processed/matched_text_sample_clean.csv")
embeddings <- read.csv("data/processed/sbert_embeddings.csv")

cat("Matched data:", dim(df), "\n")
cat("Embeddings:", dim(embeddings), "\n")

#Merge data

full_df <- df %>%
  inner_join(embeddings, by = "company_id")

cat("Merged data:", dim(full_df), "\n")
print(table(full_df$series_a_success))

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

full_df$series_a_success <- factor(ifelse(full_df$series_a_success == 1, "Yes", "No"), levels = c("No", "Yes"))

y <- full_df$series_a_success

#Evaluation helper function

calculate_metrics <- function(actual, prob, threshold = 0.5) {
  
  prob <- as.numeric(prob)
  prob[is.na(prob)] <- mean(actual == "Yes")
  
  pred <- ifelse(prob >= threshold, "Yes", "No")
  
  pred <- factor(pred, levels = c("No", "Yes"))
  actual <- factor(actual, levels = c("No", "Yes"))
  
  cm <- confusionMatrix(
    pred,
    actual,
    positive = "Yes"
  )
  
  auc <- as.numeric(
    roc(
      actual,
      prob,
      levels = c("No", "Yes"),
      quiet = TRUE
    )$auc
  )
  
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

#Cross-validation setup

folds <- createFolds(y, k = 5, returnTrain = FALSE)

all_results <- list()
all_predictions <- list()

counter <- 1
pred_counter <- 1

#Run models

for (fold_id in seq_along(folds)) {
  
  cat("Running fold", fold_id, "\n")
  
  test_idx <- folds[[fold_id]]
  train_idx <- setdiff(seq_len(nrow(full_df)), test_idx)
  
  train_df <- full_df[train_idx, ]
  test_df <- full_df[test_idx, ]
  
  y_train <- y[train_idx]
  y_test <- y[test_idx]
  
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
    
    x_train_cat[[col]] <- as.factor(x_train_cat[[col]])
    
    x_test_cat[[col]] <- factor(
      x_test_cat[[col]],
      levels = levels(x_train_cat[[col]])
    )
  }
  
  dummy_encoder <- dummyVars(~ ., data = x_train_cat)
  
  x_train_cat_encoded <- predict(
    dummy_encoder,
    newdata = x_train_cat
  )
  
  x_test_cat_encoded <- predict(
    dummy_encoder,
    newdata = x_test_cat
  )
  
  x_train_cat_encoded <- as.data.frame(x_train_cat_encoded)
  x_test_cat_encoded <- as.data.frame(x_test_cat_encoded)
  
  names(x_train_cat_encoded) <- make.names(
    names(x_train_cat_encoded),
    unique = TRUE
  )
  
  names(x_test_cat_encoded) <- make.names(
    names(x_test_cat_encoded),
    unique = TRUE
  )
  
  missing_in_test <- setdiff(
    names(x_train_cat_encoded),
    names(x_test_cat_encoded)
  )
  
  for (col in missing_in_test) {
    x_test_cat_encoded[[col]] <- 0
  }
  
  extra_in_test <- setdiff(
    names(x_test_cat_encoded),
    names(x_train_cat_encoded)
  )
  
  if (length(extra_in_test) > 0) {
    
    x_test_cat_encoded <- x_test_cat_encoded[
      ,
      !(names(x_test_cat_encoded) %in% extra_in_test)
    ]
  }
  
  x_test_cat_encoded <- x_test_cat_encoded[
    ,
    names(x_train_cat_encoded)
  ]
  
  #PCA on SBERT embeddings
  
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
  
  #cap components to reduce overfitting
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
  
  #Combine structured + text features
  
  x_train <- cbind(
    x_train_num,
    x_train_cat_encoded,
    train_pca
  )
  
  x_test <- cbind(
    x_test_num,
    x_test_cat_encoded,
    test_pca
  )
  
  x_train <- as.data.frame(x_train)
  x_test <- as.data.frame(x_test)
  
  names(x_train) <- make.names(
    names(x_train),
    unique = TRUE
  )
  
  names(x_test) <- make.names(
    names(x_test),
    unique = TRUE
  )
  
  x_test <- x_test[, names(x_train)]
  
  zero_var <- nearZeroVar(x_train)
  
  if (length(zero_var) > 0) {
    
    x_train <- x_train[
      ,
      -zero_var,
      drop = FALSE
    ]
    
    x_test <- x_test[
      ,
      names(x_train),
      drop = FALSE
    ]
  }
  
  #Remove any remaining missing values
  for (col in names(x_train)) {
    x_train[[col]] <- as.numeric(x_train[[col]])
    x_test[[col]] <- as.numeric(x_test[[col]])
    
    med <- median(x_train[[col]], na.rm = TRUE)
    if (is.na(med)) med <- 0
    
    x_train[[col]][is.na(x_train[[col]])] <- med
    x_test[[col]][is.na(x_test[[col]])] <- med
  }
  
  y_train_num <- ifelse(y_train == "Yes", 1, 0)
  
  scale_pos_weight <- sum(y_train_num == 0) /
    sum(y_train_num == 1)
  
  #Logistic Regression
  
  train_glm <- data.frame(
    series_a_success = y_train,
    x_train
  )
  
  logit_model <- suppressWarnings(
    glm(
      series_a_success ~ .,
      data = train_glm,
      family = binomial
    )
  )
  
  logit_prob <- suppressWarnings(
    predict(
      logit_model,
      newdata = x_test,
      type = "response"
    )
  )
  
  logit_prob[is.na(logit_prob)] <- mean(y_train == "Yes")
  
  result <- calculate_metrics(
    y_test,
    logit_prob
  )
  
  result$model <- "Logistic Regression"
  result$fold <- fold_id
  result$n_text_pcs <- n_components
  
  all_results[[counter]] <- result
  counter <- counter + 1
  
  all_predictions[[pred_counter]] <- data.frame(
    model = "Logistic Regression",
    fold = fold_id,
    actual = y_test,
    predicted_probability = logit_prob,
    predicted_class = ifelse(
      logit_prob >= 0.5,
      "Yes",
      "No"
    ),
    n_text_pcs = n_components
  )
  
  pred_counter <- pred_counter + 1
  
  #Random Forest
  
  rf_model <- randomForest(
    x = x_train,
    y = y_train,
    ntree = 500,
    importance = TRUE
  )
  
  rf_prob <- predict(
    rf_model,
    newdata = x_test,
    type = "prob"
  )[,"Yes"]
  
  result <- calculate_metrics(
    y_test,
    rf_prob
  )
  
  result$model <- "Random Forest"
  result$fold <- fold_id
  result$n_text_pcs <- n_components
  
  all_results[[counter]] <- result
  counter <- counter + 1
  
  all_predictions[[pred_counter]] <- data.frame(
    model = "Random Forest",
    fold = fold_id,
    actual = y_test,
    predicted_probability = rf_prob,
    predicted_class = ifelse(
      rf_prob >= 0.5,
      "Yes",
      "No"
    ),
    n_text_pcs = n_components
  )
  
  pred_counter <- pred_counter + 1
  
  #XGBoost
  
  xgb_model <- xgboost(
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
  
  xgb_prob <- predict(
    xgb_model,
    newdata = as.matrix(x_test)
  )
  
  xgb_prob <- as.numeric(xgb_prob)
  
  result <- calculate_metrics(
    y_test,
    xgb_prob
  )
  
  result$model <- "XGBoost"
  result$fold <- fold_id
  result$n_text_pcs <- n_components
  
  all_results[[counter]] <- result
  counter <- counter + 1
  
  all_predictions[[pred_counter]] <- data.frame(
    model = "XGBoost",
    fold = fold_id,
    actual = y_test,
    predicted_probability = xgb_prob,
    predicted_class = ifelse(
      xgb_prob >= 0.5,
      "Yes",
      "No"
    ),
    n_text_pcs = n_components
  )
  
  pred_counter <- pred_counter + 1
}

#Save results

text_results <- bind_rows(all_results)
text_predictions <- bind_rows(all_predictions)

text_summary <- text_results %>%
  group_by(model) %>%
  summarise(
    mean_roc_auc = mean(roc_auc, na.rm = TRUE),
    mean_accuracy = mean(accuracy, na.rm = TRUE),
    mean_precision = mean(precision, na.rm = TRUE),
    mean_recall = mean(recall, na.rm = TRUE),
    mean_f1 = mean(f1, na.rm = TRUE),
    mean_text_pcs = mean(n_text_pcs, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_roc_auc))

print(text_summary)

write.csv(
  text_results,
  "reports/models/text_enhanced_model_results_by_fold.csv",
  row.names = FALSE
)

write.csv(
  text_predictions,
  "reports/models/text_enhanced_model_predictions.csv",
  row.names = FALSE
)

write.csv(
  text_summary,
  "reports/models/text_enhanced_model_summary.csv",
  row.names = FALSE
)
