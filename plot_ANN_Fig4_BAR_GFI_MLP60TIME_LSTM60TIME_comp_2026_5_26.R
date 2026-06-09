#R
# 2026.05.26
# Plot Global feature importance comparison bar plots
# MLP60TIME & LSTM60TIME
# Figure 4 of Article

library(ggplot2)
library(dplyr)
library(tidyr)
library(ggpubr)
library(shapviz)

sim_name <- "Comprehensive"

# --- 1. LOAD DATA: MLP60TIME ---
model_name_mlp <- "MLP60TIME"
import_filename_mlp <- paste0("./Results/SHAP_data_", model_name_mlp, "_", sim_name, ".rds")
shap_export_data_mlp <- readRDS(import_filename_mlp)

raw_shap_matrix_mlp <- as.matrix(shap_export_data_mlp$shap_vals$shapley_values)
global_importance_mlp <- colMeans(abs(raw_shap_matrix_mlp))

importance_df_MLP <- data.frame(
  feature = names(global_importance_mlp),
  mean_abs_shap = as.numeric(global_importance_mlp)
) %>% arrange(mean_abs_shap) %>% mutate(feature = factor(feature, levels = feature))

# --- 2. LOAD DATA: LSTM60TIME ---
model_name_lstm <- "LSTM60TIME"
import_filename_lstm <- paste0("./Results/SHAP_data_", model_name_lstm, "_", sim_name, ".rds")
shap_export_data_lstm <- readRDS(import_filename_lstm)

raw_shap_matrix_lstm <- as.matrix(shap_export_data_lstm$shap_vals$shapley_values)
global_importance_lstm <- colMeans(abs(raw_shap_matrix_lstm))

importance_df_LSTM <- data.frame(
  feature = names(global_importance_lstm),
  mean_abs_shap = as.numeric(global_importance_lstm)
) %>% arrange(mean_abs_shap) %>% mutate(feature = factor(feature, levels = feature))

# --- 3. SYNCHRONIZE AXES FOR FAIR COMPARISON ---
# Find the absolute maximum SHAP value across BOTH models to lock the X-axes together
max_shap_val <- max(max(importance_df_MLP$mean_abs_shap), max(importance_df_LSTM$mean_abs_shap))
x_axis_limit <- max_shap_val * 1.25 # Add 25% padding so the text labels never get clipped

# --- 4. GENERATE PLOTS ---
# Plot A: MLP
p_bar_MLP <- ggplot(importance_df_MLP, aes(x = mean_abs_shap, y = feature, fill = mean_abs_shap)) +
  geom_col(color = "black", linewidth = 0.3) + 
  # Use sprintf to ensure exactly 4 decimal places for consistency
  geom_text(aes(label = sprintf("%.4f", mean_abs_shap)), hjust = -0.15, size = 4) +
  scale_fill_viridis_c(option = "rocket", direction = -1, guide = "none") +
  # Add this to explicitly control the axis expansion
  #scale_x_continuous(expand = expansion(mult = c(0, 0.03))) +
  # Explicitly define the axis breaks and limits
  scale_x_continuous(
    breaks = seq(0, 0.1, by = 0.02), # Creates ticks at 0.00, 0.03, 0.06, 0.09, 0.12
    limits = c(0, 0.12),             # Hard-locks the axis limit slightly past 0.12
    expand = c(0, 0)                  # Removes unpredictable ggplot padding
  ) +
  theme_pubr(base_size = 13) + 
  theme(
    axis.text.y = element_text(size = 12, face = "bold", color = "black"),
    axis.text.x = element_text(size = 11, color = "black"),
    plot.title = element_text(face = "bold", hjust = 0.5, size = 14),
    panel.grid.major.x = element_line(color = "grey90", linetype = "dashed")
  ) +
  labs(title = "MLP60TIME (Feedforward)", x = "Mean Absolute SHAP Value", y = NULL) 

# Plot B: LSTM
p_bar_LSTM <- ggplot(importance_df_LSTM, aes(x = mean_abs_shap, y = feature, fill = mean_abs_shap)) +
  geom_col(color = "black", linewidth = 0.3) + 
  geom_text(aes(label = sprintf("%.4f", mean_abs_shap)), hjust = -0.15, size = 4) +
  scale_fill_viridis_c(option = "rocket", direction = -1, guide = "none") +
  # Add this to explicitly control the axis expansion
  # Explicitly define the axis breaks and limits
  scale_x_continuous(
    breaks = seq(0, 0.1, by = 0.02), # Creates ticks at 0.00, 0.03, 0.06, 0.09, 0.12
    limits = c(0, 0.12),             # Hard-locks the axis limit slightly past 0.12
    expand = c(0, 0)                  # Removes unpredictable ggplot padding
  ) +
  theme_pubr(base_size = 13) + 
  theme(
    axis.text.y = element_text(size = 12, face = "bold", color = "black"),
    axis.text.x = element_text(size = 11, color = "black"),
    plot.title = element_text(face = "bold", hjust = 0.5, size = 14),
    panel.grid.major.x = element_line(color = "grey90", linetype = "dashed")
  ) +
  labs(title = "LSTM60TIME (Recurrent)", x = "Mean Absolute SHAP Value", y = NULL) 
  

# --- 5. COMBINE AND SAVE ---
Fig4_comp_bar_plot <- ggarrange(
  p_bar_MLP, p_bar_LSTM,
  ncol = 2, nrow = 1,
  labels = c("A", "B"),
  font.label = list(size = 18, face = "bold") # Clear, professional A/B tags
)

print(Fig4_comp_bar_plot)

# Save targeting exact Elsevier full-page specifications (190mm wide)
ggsave(filename = paste0("Figure4_MLP60_LSTM60_Forecast_SHAP_Bar_", sim_name, "_Elsevier.png"), 
       plot = Fig4_comp_bar_plot,
       width = 7.48, height = 5.5, dpi = 500, bg = "white")

message("Elsevier-compliant SHAP Bar plot successfully generated and saved.")

