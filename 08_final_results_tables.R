#Thesis analysis: Part 8
#Result tables
library(dplyr)

dir.create("reports/tables", recursive = TRUE, showWarnings = FALSE)

model_comparison <- read.csv("reports/models/model_comparison_summary.csv")
model_differences <- read.csv("reports/models/model_comparison_differences.csv")
statistical_comparison <- read.csv("reports/models/statistical_comparison.csv")
feature_type_summary <- read.csv("reports/models/xgboost_feature_type_summary.csv")

#Final model comparison table

final_model_comparison <- model_comparison %>%
  mutate(
    mean_roc_auc = round(mean_roc_auc, 3),
    mean_accuracy = round(mean_accuracy, 3),
    mean_precision = round(mean_precision, 3),
    mean_recall = round(mean_recall, 3),
    mean_f1 = round(mean_f1, 3)
  ) %>%
  arrange(model, feature_set)

print(final_model_comparison)

write.csv(
  final_model_comparison,
  "reports/tables/final_model_comparison_table.csv",
  row.names = FALSE
)

#Final model difference table

final_model_differences <- model_differences %>%
  mutate(
    roc_auc_difference = round(roc_auc_difference, 3),
    accuracy_difference = round(accuracy_difference, 3),
    precision_difference = round(precision_difference, 3),
    recall_difference = round(recall_difference, 3),
    f1_difference = round(f1_difference, 3)
  )

print(final_model_differences)

write.csv(
  final_model_differences,
  "reports/tables/final_model_difference_table.csv",
  row.names = FALSE
)

#Final statistical comparison table

final_statistical_comparison <- statistical_comparison %>%
  mutate(
    mean_roc_auc_text = round(mean_roc_auc_text, 3),
    mean_roc_auc_structured = round(mean_roc_auc_structured, 3),
    roc_auc_difference = round(roc_auc_difference, 3),
    roc_auc_p_value = round(roc_auc_p_value, 3),
    mean_f1_text = round(mean_f1_text, 3),
    mean_f1_structured = round(mean_f1_structured, 3),
    f1_difference = round(f1_difference, 3),
    f1_p_value = round(f1_p_value, 3)
  )

print(final_statistical_comparison)

write.csv(
  final_statistical_comparison,
  "reports/tables/final_statistical_comparison_table.csv",
  row.names = FALSE
)

#Final feature type importance table

final_feature_type_importance <- feature_type_summary %>%
  mutate(
    total_gain = round(total_gain, 3),
    gain_share = round(gain_share, 3)
  )

print(final_feature_type_importance)

write.csv(
  final_feature_type_importance,
  "reports/tables/final_feature_type_importance_table.csv",
  row.names = FALSE
)