# ==============================================================================
# Script: 08_Plot_Figs2_3_4_SHAP_Analysis.R
# Title: Master SHAP Visualization Suite (Figures 2, 3, and 4)
# Description: Generates Figure 2 (Temporal Rolling-Origin SHAP Stability),
#              Figure 3 (Spatial 10-Fold Block CV SHAP Stability), and
#              Figure 4 (Spatial Beeswarm Feature Interactions).
# Reference: Stips et al. (2026) - Environmental Modelling & Software
# License: GPL-3.0
# ==============================================================================

library(ggplot2)
library(dplyr)
library(ggpubr)
library(shapviz)
library(patchwork)

# --- 1. DIRECTORY SETUP & INPUT CHECK ---
in_dir  <- ifelse(dir.exists("./Output"), "./Output", "./")
fig_dir <- "./Figures"
if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)

sum_mlp_spat  <- file.path(in_dir, "Summary_MLP_SPAT.rds")
sum_lstm_spat <- file.path(in_dir, "Summary_LSTM_SPAT.rds")
sum_mlp_time  <- file.path(in_dir, "Summary_MLP_TIME.rds")
sum_lstm_time <- file.path(in_dir, "Summary_LSTM_TIME.rds")
raw_mlp_spat  <- file.path(in_dir, "SHAP_RAW_MLP_SPAT.rds")
raw_lstm_spat <- file.path(in_dir, "SHAP_RAW_LSTM_SPAT.rds")

# ==============================================================================
# 2. FIGURE 2: TEMPORAL STABILITY OF GLOBAL FEATURE IMPORTANCE (30-YR ROLLING)
# ==============================================================================
mlp_time_data  <- readRDS(sum_mlp_time)
lstm_time_data <- readRDS(sum_lstm_time)

mlp_time_shap  <- mlp_time_data$final_shap  %>% filter(Scenario == "Comprehensive")
lstm_time_shap <- lstm_time_data$final_shap %>% filter(Scenario == "Comprehensive")
mlp_time_raw   <- mlp_time_data$raw_shap    %>% filter(Scenario == "Comprehensive")
lstm_time_raw  <- lstm_time_data$raw_shap   %>% filter(Scenario == "Comprehensive")

# Harmonize feature order by MLP_TIME importance
ordered_features_time <- rev(mlp_time_shap$Feature)

mlp_time_shap$Feature  <- factor(mlp_time_shap$Feature,  levels = ordered_features_time)
lstm_time_shap$Feature <- factor(lstm_time_shap$Feature, levels = ordered_features_time)
mlp_time_shap$Model    <- "MLP_TIME (Feedforward)"
lstm_time_shap$Model   <- "LSTM_TIME (Recurrent)"
combined_time_shap     <- bind_rows(mlp_time_shap, lstm_time_shap)

mlp_time_raw$Feature   <- factor(mlp_time_raw$Feature,   levels = ordered_features_time)
lstm_time_raw$Feature  <- factor(lstm_time_raw$Feature,  levels = ordered_features_time)
mlp_time_raw$Model     <- "MLP_TIME (Feedforward)"
lstm_time_raw$Model    <- "LSTM_TIME (Recurrent)"
combined_time_raw      <- bind_rows(mlp_time_raw, lstm_time_raw)

max_x_limit <- max(0.12, max(combined_time_raw$SHAP_Mean_Abs) * 1.05)

p_fig2 <- ggplot(combined_time_shap, aes(x = Feature, y = SHAP_Mean, fill = Model)) +
  geom_col(color = "black", linewidth = 0.3, width = 0.75, alpha = 0.85) +
  geom_errorbar(aes(ymin = pmax(0, SHAP_Mean - SHAP_CI), ymax = SHAP_Mean + SHAP_CI),
                width = 0.25, color = "black", linewidth = 0.6) +
  geom_point(data = combined_time_raw, aes(x = Feature, y = SHAP_Mean_Abs), 
             position = position_jitter(width = 0.15, height = 0, seed = 123),
             color = "black", size = 1.6, alpha = 0.65, inherit.aes = FALSE) +
  coord_flip(ylim = c(0, max_x_limit)) +
  facet_wrap(~Model, scales = "fixed") + 
  scale_fill_manual(values = c("MLP_TIME (Feedforward)" = "#E69F00", 
                               "LSTM_TIME (Recurrent)"  = "#56B4E9")) +
  labs(
    title    = "Temporal Stability of Global Feature Importance (30-Year Rolling Forecast)",
    subtitle = "Comprehensive Variable Suite (Means \u00B1 95% CI with individual forecast origins overlaid)",
    x        = NULL,
    y        = "Mean Absolute SHAP Value (Impact on TTI Forecast)"
  ) +
  theme_pubr(base_size = 14) +
  theme(
    plot.title         = element_text(face = "bold", hjust = 0.5, size = 14.5),
    plot.subtitle      = element_text(hjust = 0.5, size = 12, color = "grey30", margin = margin(b = 15)),
    strip.text         = element_text(face = "bold", size = 14, margin = margin(b = 10)),
    strip.background   = element_blank(),
    legend.position    = "none", 
    panel.grid.major.y = element_blank(), 
    panel.grid.major.x = element_line(color = "grey90", linetype = "dashed"), 
    axis.text.y        = element_text(face = "bold", color = "black", size = 12),
    axis.text.x        = element_text(color = "black", size = 11),
    axis.title.x       = element_text(face = "bold", margin = margin(t = 12)),
    panel.spacing      = unit(2, "lines") 
  )

fig2_file <- file.path(fig_dir, "Figure2_Temporal_SHAP_Stability.png")
ggsave(fig2_file, plot = p_fig2, width = 8.5, height = 5.5, dpi = 500, bg = "white")
message("Figure 2 successfully saved to: ", fig2_file)

# ==============================================================================
# 3. FIGURE 3: SPATIAL STABILITY OF GLOBAL FEATURE IMPORTANCE (10-FOLD CV)
# ==============================================================================
mlp_spat_data  <- readRDS(sum_mlp_spat)
lstm_spat_data <- readRDS(sum_lstm_spat)

mlp_spat_sum  <- mlp_spat_data$final_shap  %>% filter(Scenario == "Comprehensive")
lstm_spat_sum <- lstm_spat_data$final_shap %>% filter(Scenario == "Comprehensive")
mlp_spat_raw  <- mlp_spat_data$raw_shap    %>% filter(Scenario == "Comprehensive")
lstm_spat_raw <- lstm_spat_data$raw_shap   %>% filter(Scenario == "Comprehensive")

ordered_features_spat <- rev(mlp_spat_sum$Feature)

mlp_spat_sum$Feature  <- factor(mlp_spat_sum$Feature,  levels = ordered_features_spat)
lstm_spat_sum$Feature <- factor(lstm_spat_sum$Feature, levels = ordered_features_spat)
mlp_spat_sum$Model    <- "MLP_SPAT (Feedforward)"
lstm_spat_sum$Model   <- "LSTM_SPAT (Recurrent)"
combined_spat_sum     <- bind_rows(mlp_spat_sum, lstm_spat_sum)

mlp_spat_raw$Feature  <- factor(mlp_spat_raw$Feature,  levels = ordered_features_spat)
lstm_spat_raw$Feature <- factor(lstm_spat_raw$Feature, levels = ordered_features_spat)
mlp_spat_raw$Model    <- "MLP_SPAT (Feedforward)"
lstm_spat_raw$Model   <- "LSTM_SPAT (Recurrent)"
combined_spat_raw     <- bind_rows(mlp_spat_raw, lstm_spat_raw)

p_fig3 <- ggplot(combined_spat_sum, aes(x = Feature, y = SHAP_Mean, fill = Model)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7, 
           color = "black", linewidth = 0.3, alpha = 0.85) +
  geom_errorbar(aes(ymin = pmax(0, SHAP_Mean - SHAP_CI), ymax = SHAP_Mean + SHAP_CI),
                position = position_dodge(width = 0.8), width = 0.25, 
                color = "black", linewidth = 0.7) +
  geom_point(data = combined_spat_raw, aes(x = Feature, y = SHAP_Mean_Abs, fill = Model), 
             position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.8, seed = 123),
             color = "black", shape = 21, size = 1.5, alpha = 0.7, show.legend = FALSE) +
  coord_flip() +
  scale_fill_manual(values = c("MLP_SPAT (Feedforward)" = "#E69F00", 
                               "LSTM_SPAT (Recurrent)"  = "#56B4E9")) +
  labs(
    title    = "Spatial Stability of Global Feature Importance (10-Fold Block Cross-Validation)",
    subtitle = "10-Fold Spatial Block Cross-Validation (Means \u00B1 95% CI with fold-level iterations)",
    x        = NULL,
    y        = "Mean Absolute SHAP Value (Impact on TTI)",
    fill     = "ANN Architecture:"
  ) +
  theme_pubr(base_size = 14) +
  theme(
    plot.title         = element_text(face = "bold", hjust = 0.5, size = 13.5),
    plot.subtitle      = element_text(hjust = 0.5, size = 12, color = "grey30", margin = margin(b = 15)),
    legend.position    = "bottom",
    legend.title       = element_text(face = "bold"),
    legend.text        = element_text(size = 12),
    legend.background  = element_rect(fill = NA, color = NA), 
    legend.margin      = margin(t = 10),
    panel.grid.major.y = element_blank(), 
    panel.grid.major.x = element_line(color = "grey90", linetype = "dashed"), 
    axis.text.y        = element_text(face = "bold", color = "black", size = 12),
    axis.text.x        = element_text(color = "black", size = 11),
    axis.title.x       = element_text(face = "bold", margin = margin(t = 12)),
    axis.line          = element_line(linewidth = 0.8) 
  )

fig3_file <- file.path(fig_dir, "Figure3_Spatial_SHAP_Stability.png")
ggsave(fig3_file, plot = p_fig3, width = 7.8, height = 5.5, dpi = 500, bg = "white")
message("Figure 3 successfully saved to: ", fig3_file)

# ==============================================================================
# 4. FIGURE 4: SPATIAL SHAP BEESWARM DISTRIBUTION (MLP_SPAT vs LSTM_SPAT)
# ==============================================================================
mlp_raw_export  <- readRDS(raw_mlp_spat)
lstm_raw_export <- readRDS(raw_lstm_spat)

# Extract Representative Test Fold 1 (N = 2000 pixels) as documented in Figure 4 caption
n_rep <- min(2000, nrow(mlp_raw_export$Comprehensive$shap_vals))
shp_mlp <- shapviz(
  mlp_raw_export$Comprehensive$shap_vals[1:n_rep, ], 
  X = mlp_raw_export$Comprehensive$feature_data[1:n_rep, ]
)
shp_lstm <- shapviz(
  lstm_raw_export$Comprehensive$shap_vals[1:n_rep, ], 
  X = lstm_raw_export$Comprehensive$feature_data[1:n_rep, ]
)

p_mlp_bee <- sv_importance(shp_mlp, kind = "beeswarm", max_display = 7, show_numbers = TRUE) + 
  scale_color_viridis_c(option = "viridis", name = "Feature\nValue") +
  theme_pubr(base_size = 14) + 
  labs(
    title = "A: MLP_SPAT (Feedforward)",
    x     = "SHAP Value (Impact on TTI)",
    y     = "Environmental Predictors"
  ) +
  theme(
    plot.title         = element_text(face = "bold", hjust = 0, size = 15),
    axis.text.y        = element_text(face = "bold", color = "black", size = 12),
    axis.text.x        = element_text(color = "black", size = 11),
    axis.title         = element_text(face = "bold", size = 13),
    panel.grid.major.x = element_line(color = "grey80", linetype = "dashed"),
    legend.position    = "none"
  )

p_lstm_bee <- sv_importance(shp_lstm, kind = "beeswarm", max_display = 7, show_numbers = TRUE) + 
  scale_color_viridis_c(option = "viridis", name = "Feature\nValue") +
  theme_pubr(base_size = 14) + 
  labs(
    title = "B: LSTM_SPAT (Recurrent)",
    x     = "SHAP Value (Impact on TTI)",
    y     = NULL 
  ) +
  theme(
    plot.title         = element_text(face = "bold", hjust = 0, size = 15),
    axis.text.y        = element_text(face = "bold", color = "black", size = 12),
    axis.text.x        = element_text(color = "black", size = 11),
    axis.title         = element_text(face = "bold", size = 13),
    panel.grid.major.x = element_line(color = "grey80", linetype = "dashed"),
    legend.title       = element_text(face = "bold", size = 12, angle = 90, hjust = 0.5),
    legend.text        = element_text(size = 10),
    legend.position    = "right"
  )

p_fig4 <- p_mlp_bee + p_lstm_bee + 
  plot_annotation(
    title = "Spatial SHAP Distribution: Comprehensive Suite",
    theme = theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 17, margin = margin(b = 15)))
  )

fig4_file <- file.path(fig_dir, "Figure4_Spatial_SHAP_Beeswarm.png")

ggsave(fig4_file, plot = p_fig4, width = 11, height = 6, dpi = 500, bg = "white")

message("Figure 4 successfully saved to: ", fig4_file)

