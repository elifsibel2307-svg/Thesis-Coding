# Thesis analysis Part 2
# Structured-only baseline models

set.seed(123)

library(dplyr)
library(caret)
library(pROC)
library(randomForest)
library(xgboost)

dir.create("reports/models", recursive = TRUE, showWarnings = FALSE)

#Load cleaned matched data

df <- read.csv("data/processed/matched_text_sample_clean.csv")

cat("Rows:", nrow(df), "\n")
cat("Success balance:\n")
print(table(df$series_a_success))

#Select structured features

numeric_features <- c(
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

model_df <- df %>%
  select(series_a_success, all_of(numeric_features), all_of(categorical_features))

#Handle missing values


for (col in numeric_features) {
  model_df[[col]] <- as.numeric(model_df[[col]])
  model_df[[col]][is.na(model_df[[col]])] <- median(model_df[[col]], na.rm = TRUE)
}

for (col in categorical_features) {
  model_df[[col]] <- as.character(model_df[[col]])
  model_df[[col]][is.na(model_df[[col]]) | model_df[[col]] == ""] <- "Unknown"
  model_df[[col]] <- as.factor(model_df[[col]])
}

model_df$series_a_success <- as.factor(model_df$series_a_success)
levels(model_df$series_a_success) <- c("No", "Yes")

#Encode categorical variables

predictor_df <- model_df %>% select(-series_a_success)

dummy_encoder <- dummyVars(~ ., data = predictor_df)

x <- predict(dummy_encoder, newdata = predictor_df)
x <- as.data.frame(x)

#Make column names safe for glm
names(x) <- make.names(names(x), unique = TRUE)

y <- model_df$series_a_success

#remove zero-variance columns
zero_var <- nearZeroVar(x)
if (length(zero_var) > 0) {
  x <- x[, -zero_var]
}

#Evaluation helper function

calculate_metrics <- function(actual, prob, threshold = 0.5) {
  pred <- ifelse(prob >= threshold, "Yes", "No")
  pred <- factor(pred, levels = c("No", "Yes"))
  actual <- factor(actual, levels = c("No", "Yes"))
  
  cm <- confusionMatrix(pred, actual, positive = "Yes")
  
  auc <- as.numeric(roc(actual, prob, levels = c("No", "Yes"), quiet = TRUE)$auc)
  
  data.frame(
    roc_auc = auc,
    accuracy = as.numeric(cm$overall["Accuracy"]),
    precision = as.numeric(cm$byClass["Precision"]),
    recall = as.numeric(cm$byClass["Recall"]),
    f1 = as.numeric(cm$byClass["F1"]),
    tp = cm$table["Yes", "Yes"],
    tn = cm$table["No", "No"],
    fp = cm$table["Yes", "No"],
    fn = cm$table["No", "Yes"]
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
  train_idx <- setdiff(seq_len(nrow(x)), test_idx)
  
  x_train <- x[train_idx, ]
  x_test <- x[test_idx, ]
  
  y_train <- y[train_idx]
  y_test <- y[test_idx]
  
  # class weights for imbalance
  y_train_num <- ifelse(y_train == "Yes", 1, 0)
  scale_pos_weight <- sum(y_train_num == 0) / sum(y_train_num == 1)
  
  #Dummy baseline
  dummy_prob <- rep(mean(y_train == "Yes"), length(y_test))
  
  result <- calculate_metrics(y_test, dummy_prob)
  result$model <- "Dummy baseline"
  result$fold <- fold_id
  all_results[[counter]] <- result
  counter <- counter + 1
  
  all_predictions[[pred_counter]] <- data.frame(
    model = "Dummy baseline",
    fold = fold_id,
    actual = y_test,
    predicted_probability = dummy_prob,
    predicted_class = ifelse(dummy_prob >= 0.5, "Yes", "No")
  )
  pred_counter <- pred_counter + 1
  
  #Logistic Regression
  train_glm <- data.frame(series_a_success = y_train, x_train)
  
  logit_model <- glm(
    series_a_success ~ .,
    data = train_glm,
    family = binomial
  )
  
  logit_prob <- predict(
    logit_model,
    newdata = x_test,
    type = "response"
  )
  
  result <- calculate_metrics(y_test, logit_prob)
  result$model <- "Logistic Regression"
  result$fold <- fold_id
  all_results[[counter]] <- result
  counter <- counter + 1
  
  all_predictions[[pred_counter]] <- data.frame(
    model = "Logistic Regression",
    fold = fold_id,
    actual = y_test,
    predicted_probability = logit_prob,
    predicted_class = ifelse(logit_prob >= 0.5, "Yes", "No")
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
  
  result <- calculate_metrics(y_test, rf_prob)
  result$model <- "Random Forest"
  result$fold <- fold_id
  all_results[[counter]] <- result
  counter <- counter + 1
  
  all_predictions[[pred_counter]] <- data.frame(
    model = "Random Forest",
    fold = fold_id,
    actual = y_test,
    predicted_probability = rf_prob,
    predicted_class = ifelse(rf_prob >= 0.5, "Yes", "No")
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
  
  result <- calculate_metrics(y_test, xgb_prob)
  result$model <- "XGBoost"
  result$fold <- fold_id
  all_results[[counter]] <- result
  counter <- counter + 1
  
  all_predictions[[pred_counter]] <- data.frame(
    model = "XGBoost",
    fold = fold_id,
    actual = y_test,
    predicted_probability = xgb_prob,
    predicted_class = ifelse(xgb_prob >= 0.5, "Yes", "No")
  )
  pred_counter <- pred_counter + 1
}

#Save results
structured_results <- bind_rows(all_results)
structured_predictions <- bind_rows(all_predictions)

structured_summary <- structured_results %>%
  group_by(model) %>%
  summarise(
    mean_roc_auc = mean(roc_auc, na.rm = TRUE),
    mean_accuracy = mean(accuracy, na.rm = TRUE),
    mean_precision = mean(precision, na.rm = TRUE),
    mean_recall = mean(recall, na.rm = TRUE),
    mean_f1 = mean(f1, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_roc_auc))

print(structured_summary)

write.csv(
  structured_results,
  "reports/models/structured_model_results_by_fold.csv",
  row.names = FALSE
)

write.csv(
  structured_predictions,
  "reports/models/structured_model_predictions.csv",
  row.names = FALSE
)

write.csv(
  structured_summary,
  "reports/models/structured_model_summary.csv",
  row.names = FALSE
)