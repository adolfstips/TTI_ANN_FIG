#R

# 2026.05.25

# Plot Figure 2 - K-fold cross validation results.

library(ggplot2)
library(dplyr)
library(ggpubr) # Added for Elsevier publication theme

# --- 1. PREPARE THE COMPREHENSIVE K-FOLD SHAP DATA ---
shap_data <- data.frame(
  Feature = rep(c("Depth", "Secchi", "Chla", "Pho", "Nit", "Temp", "DO2_mg"), 2),
  Model   = c(rep("MLP70SPAT (Feedforward)", 7), rep("LSTM70SPAT (Recurrent)", 7)),
  Mean    = c(0.0417, 0.0370, 0.0224, 0.0224, 0.0177, 0.0110, 0.0169,  # MLP Means
              0.0478, 0.0481, 0.0390, 0.0228, 0.0207, 0.0141, 0.0134), # LSTM Means
  CI      = c(0.0061, 0.0044, 0.0016, 0.0018, 0.0011, 0.0005, 0.0025,  # MLP CIs
              0.0045, 0.0072, 0.0053, 0.0013, 0.0015, 0.0007, 0.0012)  # LSTM CIs
)

# Reorder the features dynamically based on overall dominance
shap_data$Feature <- factor(shap_data$Feature, 
                            levels = rev(c("Secchi", "Depth", "Chla", "Pho", "Nit", "DO2_mg", "Temp")))

# --- 2. GENERATE THE TARGETED PLOT ---
p_robustness <- ggplot(shap_data, aes(x = Feature, y = Mean, fill = Model)) +
  # Create the side-by-side comparison bar layout
  geom_col(position = position_dodge(width = 0.8), width = 0.7, color = "black", linewidth = 0.3) +
  
  # Add the 95% Confidence Interval Error Bars
  geom_errorbar(aes(ymin = Mean - CI, ymax = Mean + CI),
                position = position_dodge(width = 0.8), width = 0.3, 
                color = "black", linewidth = 0.7) +
  
  coord_flip() +
  
  scale_fill_manual(values = c("MLP70SPAT (Feedforward)" = "#fee08b", 
                               "LSTM70SPAT (Recurrent)" = "#5e3c99")) +
  
  labs(
    title = "Statistical Stability of Global Feature Importance",
    subtitle = "10-Fold Spatial Block Cross-Validation (Comprehensive Variable Suite)",
    x = NULL,
    y = "Mean Absolute SHAP Value ± 95% CI (Impact on TTI)",
    fill = "ANN Architecture:"
  ) +
  
  # Deploying ggpubr theme for clean, solid axes
  theme_pubr(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 16),
    plot.subtitle = element_text(hjust = 0.5, size = 12, color = "grey30", margin = margin(b = 15)),
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    legend.text = element_text(size = 11),
    legend.background = element_rect(fill = NA, color = NA), # Clean legend background
    legend.margin = margin(t = 10),
    panel.grid.major.y = element_blank(), 
    panel.grid.major.x = element_line(color = "grey90", linetype = "dashed"), # Keep faint grid for reading values
    axis.text.y = element_text(face = "bold", color = "black", size = 12),
    axis.text.x = element_text(color = "black", size = 11),
    axis.title.x = element_text(face = "bold", margin = margin(t = 12)),
    axis.line = element_line(linewidth = 0.8) # Slightly thicker L-shape axes for print contrast
  )

print(p_robustness)

# Save targeting exact Elsevier full-page specifications (190mm wide)
ggsave("Figure2_Spatial_SHAP_Robustness_Elsevier.png", plot = p_robustness, 
       width = 7.48, height = 5.5, dpi = 500, bg = "white")

