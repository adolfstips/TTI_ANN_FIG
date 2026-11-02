# ==============================================================================
# Script: 06_Generate_Manuscript_Tables.R
# Title: Master Script to Generate Manuscript Tables 2, 3, and 4
# Description: Loads the harmonized Summary_*.rds outputs from Scripts 01-04
#              and generates publication-ready Word (.docx) and CSV (.csv) tables
#              for Spatial CV (Table 2), Temporal Forecasting (Table 3), and
#              Global Feature Importance (Table 4).
# Reference: Stips et al. (2026) - Environmental Modelling & Software
# License: GPL-3.0
# ==============================================================================

library(dplyr)
library(purrr)
library(gt)

# --- 1. DIRECTORY SETUP & INPUT LOADING ---
in_dir  <- ifelse(dir.exists("./Output"), "./Output", "./")
out_dir <- "./Tables"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

mlp_spat_file  <- file.path(in_dir, "Summary_MLP_SPAT.rds")
lstm_spat_file <- file.path(in_dir, "Summary_LSTM_SPAT.rds")
mlp_time_file  <- file.path(in_dir, "Summary_MLP_TIME.rds")
lstm_time_file <- file.path(in_dir, "Summary_LSTM_TIME.rds")

# Verify all 4 summary files exist before proceeding
required_files <- c(mlp_spat_file, lstm_spat_file, mlp_time_file, lstm_time_file)
if (!all(file.exists(required_files))) {
  stop("Missing one or more Summary_*.rds files in: ", in_dir,
       "\nPlease run scripts 01 to 04 first.")
}

res_mlp_spat  <- readRDS(mlp_spat_file)
res_lstm_spat <- readRDS(lstm_spat_file)
res_mlp_time  <- readRDS(mlp_time_file)
res_lstm_time <- readRDS(lstm_time_file)

# Define consistent scenario ordering across performance tables
scenario_order <- c("Comprehensive", "MSFD_Baseline", "Remote_Sensing")

# ==============================================================================
# 2. GENERATE TABLE 2: SPATIAL EXTRAPOLATION PERFORMANCE (SECTION 3.1)
# ==============================================================================
df_spat <- bind_rows(
  res_mlp_spat$metrics  %>% mutate(Architecture = "MLP_SPAT"),
  res_lstm_spat$metrics %>% mutate(Architecture = "LSTM_SPAT")
) %>%
  mutate(
    Scenario = factor(Scenario, levels = scenario_order),
    Accuracy = sprintf("%.3f (\u00B1%.3f)", ACC_Mean, ACC_CI),
    F1_Score = sprintf("%.3f (\u00B1%.3f)", F1_Mean, F1_CI),
    RMSE     = sprintf("%.3f (\u00B1%.3f)", RMSE_Mean, RMSE_CI),
    R2       = sprintf("%.3f (\u00B1%.3f)", R2_Mean, R2_CI)
  ) %>%
  arrange(Architecture, Scenario) %>%
  select(Architecture, Scenario, Accuracy, F1_Score, RMSE, R2)

cat("\n==================================================================\n")
cat("TABLE 2: SPATIAL EXTRAPOLATION PERFORMANCE (SECTION 3.1)\n")
cat("==================================================================\n")
print(df_spat, width = Inf)

tbl_2_gt <- df_spat %>%
  gt(groupname_col = "Architecture") %>%
  tab_header(
    title    = "Table 2: Spatial Extrapolation Performance",
    subtitle = "10-Fold Spatial Block Cross-Validation (Means \u00B1 95% CI)"
  ) %>%
  cols_label(
    Scenario = "Scenario",
    Accuracy = "Accuracy",
    F1_Score = "F1-Score",
    RMSE     = "RMSE",
    R2       = "R\u00B2"
  ) %>%
  cols_align(align = "center", columns = c("Accuracy", "F1_Score", "RMSE", "R2")) %>%
  tab_options(
    row_group.font.weight     = "bold",
    heading.title.font.weight = "bold",
    column_labels.font.weight = "bold"
  )

gtsave(tbl_2_gt, file.path(out_dir, "Table_2_Spatial_Performance.docx"))
write.csv(df_spat, file.path(out_dir, "Table_2_Spatial_Performance.csv"), row.names = FALSE)

# ==============================================================================
# 3. GENERATE TABLE 3: TEMPORAL FORECASTING PERFORMANCE (SECTION 3.2)
# ==============================================================================
df_time <- bind_rows(
  res_mlp_time$metrics  %>% mutate(Architecture = "MLP_TIME"),
  res_lstm_time$metrics %>% mutate(Architecture = "LSTM_TIME")
) %>%
  mutate(
    Scenario = factor(Scenario, levels = scenario_order),
    Accuracy = sprintf("%.3f (\u00B1%.3f)", ACC_Mean, ACC_CI),
    F1_Score = sprintf("%.3f (\u00B1%.3f)", F1_Mean, F1_CI),
    RMSE     = sprintf("%.3f (\u00B1%.3f)", RMSE_Mean, RMSE_CI),
    R2       = sprintf("%.3f (\u00B1%.3f)", R2_Mean, R2_CI)
  ) %>%
  arrange(Architecture, Scenario) %>%
  select(Architecture, Scenario, Accuracy, F1_Score, RMSE, R2)

cat("\n==================================================================\n")
cat("TABLE 3: TEMPORAL FORECASTING PERFORMANCE (SECTION 3.2)\n")
cat("==================================================================\n")
print(df_time, width = Inf)

tbl_3_gt <- df_time %>%
  gt(groupname_col = "Architecture") %>%
  tab_header(
    title    = "Table 3: Temporal Forecasting Performance",
    subtitle = "30-Year Rolling-Origin Hindcast (Means \u00B1 95% CI across 6 forward pentads)"
  ) %>%
  cols_label(
    Scenario = "Scenario",
    Accuracy = "Accuracy",
    F1_Score = "F1-Score",
    RMSE     = "RMSE",
    R2       = "R\u00B2"
  ) %>%
  cols_align(align = "center", columns = c("Accuracy", "F1_Score", "RMSE", "R2")) %>%
  tab_options(
    row_group.font.weight     = "bold",
    heading.title.font.weight = "bold",
    column_labels.font.weight = "bold"
  )

gtsave(tbl_3_gt, file.path(out_dir, "Table_3_Temporal_Performance.docx"))
write.csv(df_time, file.path(out_dir, "Table_3_Temporal_Performance.csv"), row.names = FALSE)

# ==============================================================================
# 4. GENERATE TABLE 4: GLOBAL FEATURE IMPORTANCE - SHAP (SECTION 3.3)
# ==============================================================================
format_shap_col <- function(res_obj, col_name) {
  res_obj$final_shap %>%
    filter(Scenario == "Comprehensive") %>%
    mutate(!!col_name := sprintf("%.4f (\u00B1%.4f)", SHAP_Mean, SHAP_CI)) %>%
    select(Feature, !!col_name)
}

shap_mlp_spat  <- format_shap_col(res_mlp_spat,  "MLP_SPAT")
shap_lstm_spat <- format_shap_col(res_lstm_spat, "LSTM_SPAT")
shap_mlp_time  <- format_shap_col(res_mlp_time,  "MLP_TIME")
shap_lstm_time <- format_shap_col(res_lstm_time, "LSTM_TIME")

# Sort rows by baseline MLP_SPAT mean importance
sort_order <- res_mlp_spat$final_shap %>%
  filter(Scenario == "Comprehensive") %>%
  select(Feature, Sort_Val = SHAP_Mean)

df_shap <- shap_mlp_spat %>%
  left_join(shap_lstm_spat, by = "Feature") %>%
  left_join(shap_mlp_time,  by = "Feature") %>%
  left_join(shap_lstm_time, by = "Feature") %>%
  left_join(sort_order,     by = "Feature") %>%
  arrange(desc(Sort_Val)) %>%
  select(-Sort_Val)

cat("\n==================================================================\n")
cat("TABLE 4: GLOBAL FEATURE IMPORTANCE - SHAP (SECTION 3.3)\n")
cat("==================================================================\n")
print(df_shap, width = Inf)

tbl_4_gt <- df_shap %>%
  gt() %>%
  tab_header(
    title    = "Table 4: Global Feature Importance (Comprehensive Scenario)",
    subtitle = "Mean Absolute SHAP value (\u00B1 95% CI) representing the impact on the continuous TTI prediction across spatial and temporal tasks."
  ) %>%
  cols_align(align = "center", columns = c("MLP_SPAT", "LSTM_SPAT", "MLP_TIME", "LSTM_TIME")) %>%
  tab_options(
    heading.title.font.weight = "bold",
    column_labels.font.weight = "bold"
  )

gtsave(tbl_4_gt, file.path(out_dir, "Table_4_SHAP_Feature_Importance.docx"))
write.csv(df_shap, file.path(out_dir, "Table_4_SHAP_Feature_Importance.csv"), row.names = FALSE)

message("\nAll manuscript tables (Tables 2, 3, and 4) successfully saved to: ", out_dir)

