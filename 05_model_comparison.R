#Thesis analysis: Part 5
#Compare structured-only and text-enhanced models

library(dplyr)

structured <- read.csv("reports/models/structured_model_summary.csv")
text <- read.csv("reports/models/text_enhanced_model_summary.csv")

structured <- structured %>%
  filter(model != "Dummy baseline") %>%
  mutate(feature_set = "Structured only")

text <- text %>%
  mutate(feature_set = "Structured + SBERT")

model_comparison <- bind_rows(structured, text) %>%
  select(
    feature_set,
    model,
    mean_roc_auc,
    mean_accuracy,
    mean_precision,
    mean_recall,
    mean_f1
  ) %>%
  arrange(model, feature_set)

print(model_comparison)

write.csv(
  model_comparison,
  "reports/models/model_comparison_summary.csv",
  row.names = FALSE
)

# Difference table: text-enhanced minus structured-only
structured_short <- structured %>%
  select(model, mean_roc_auc, mean_accuracy, mean_precision, mean_recall, mean_f1)

text_short <- text %>%
  select(model, mean_roc_auc, mean_accuracy, mean_precision, mean_recall, mean_f1)

model_differences <- text_short %>%
  inner_join(
    structured_short,
    by = "model",
    suffix = c("_text", "_structured")
  ) %>%
  mutate(
    roc_auc_difference = mean_roc_auc_text - mean_roc_auc_structured,
    accuracy_difference = mean_accuracy_text - mean_accuracy_structured,
    precision_difference = mean_precision_text - mean_precision_structured,
    recall_difference = mean_recall_text - mean_recall_structured,
    f1_difference = mean_f1_text - mean_f1_structured
  ) %>%
  select(
    model,
    roc_auc_difference,
    accuracy_difference,
    precision_difference,
    recall_difference,
    f1_difference
  )

print(model_differences)

write.csv(
  model_differences,
  "reports/models/model_comparison_differences.csv",
  row.names = FALSE
)