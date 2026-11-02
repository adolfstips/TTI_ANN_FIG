# ==============================================================================
# Script: 10_Plot_FigsS1_S2_Confusion_Matrices.R
# Title: Generate Supplementary Figures S1 and S2 (Global Confusion Matrices)
# Description: Plots the global confusion matrices aggregated across all 10
#              independent spatial cross-validation test blocks for MLP_SPAT
#              (Figure S1) and LSTM_SPAT (Figure S2).
# Reference: Stips et al. (2026) - Environmental Modelling & Software
# License: GPL-3.0
# ==============================================================================

library(cvms)
library(tibble)
library(ggplot2)
library(ggpubr)

# --- 1. DIRECTORY SETUP ---
in_dir  <- ifelse(dir.exists("./Output"), "./Output", "./")
fig_dir <- "./Figures"
if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)

# Helper function to generate standardized Confusion Matrix plots
generate_cm_plot <- function(model_name, fig_label) {
  cf_file <- file.path(in_dir, paste0("CF_Matrix_", model_name, ".rds"))
  if (!file.exists(cf_file)) stop("Confusion matrix file not found: ", cf_file)
  
  cf_matrix  <- readRDS(cf_file)
  total_pred <- sum(cf_matrix$N)
  
  p_cm <- plot_confusion_matrix(cf_matrix) +
    theme_pubr(base_size = 14) +
    labs(
      title    = paste0("Global Confusion Matrix (", model_name, " - Comprehensive)"),
      subtitle = "Aggregated across all 10 independent spatial cross-validation test blocks.",
      caption  = paste("Total Predictions:", total_pred)
    ) +
    theme(
      plot.title    = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 11, color = "grey30")
    )
  
  out_file <- file.path(fig_dir, paste0(fig_label, "_Confusion_Matrix_", model_name, ".png"))
  ggsave(out_file, plot = p_cm, width = 7, height = 6, dpi = 500, bg = "white")
  message(fig_label, " (", model_name, ") successfully saved to: ", out_file)
  return(p_cm)
}

# --- 2. GENERATE SUPPLEMENTARY FIGURES S1 & S2 ---
fig_s1 <- generate_cm_plot("MLP_SPAT",  "Figure_S1")
fig_s2 <- generate_cm_plot("LSTM_SPAT", "Figure_S2")

