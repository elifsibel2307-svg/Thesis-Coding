#Thesis analysis: Part 6
#Statistical comparison of structured vs text-enhanced models

library(dplyr)

structured_results <- read.csv("reports/models/structured_model_results_by_fold.csv")
text_results <- read.csv("reports/models/text_enhanced_model_results_by_fold.csv")

structured_results <- structured_results %>%
  filter(model != "Dummy baseline")

comparison_data <- text_results %>%
  inner_join(
    structured_results,
    by = c("model", "fold"),
    suffix = c("_text", "_structured")
  )

run_paired_tests <- function(model_name) {
  
  model_data <- comparison_data %>%
    filter(model == model_name)
  
  roc_test <- t.test(
    model_data$roc_auc_text,
    model_data$roc_auc_structured,
    paired = TRUE
  )
  
  f1_test <- t.test(
    model_data$f1_text,
    model_data$f1_structured,
    paired = TRUE
  )
  
  data.frame(
    model = model_name,
    mean_roc_auc_text = mean(model_data$roc_auc_text, na.rm = TRUE),
    mean_roc_auc_structured = mean(model_data$roc_auc_structured, na.rm = TRUE),
    roc_auc_difference = mean(model_data$roc_auc_text - model_data$roc_auc_structured, na.rm = TRUE),
    roc_auc_p_value = roc_test$p.value,
    mean_f1_text = mean(model_data$f1_text, na.rm = TRUE),
    mean_f1_structured = mean(model_data$f1_structured, na.rm = TRUE),
    f1_difference = mean(model_data$f1_text - model_data$f1_structured, na.rm = TRUE),
    f1_p_value = f1_test$p.value
  )
}

statistical_comparison <- bind_rows(
  run_paired_tests("Logistic Regression"),
  run_paired_tests("Random Forest"),
  run_paired_tests("XGBoost")
)

print(statistical_comparison)

write.csv(
  statistical_comparison,
  "reports/models/statistical_comparison.csv",
  row.names = FALSE
)