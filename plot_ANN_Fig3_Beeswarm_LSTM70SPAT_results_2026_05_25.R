#R

# make the SHAP plot LSTM70SPAT


library(ggplot2)
library(shapviz)
library(ggpubr) # For the Elsevier theme


sim_name <- "Comprehensive" # Or loop through your scenarios

# SHAP_data_LSTM70SPAT_Comprehensive.rds

import_filename <- paste0("SHAP_data_LSTM70SPAT_", sim_name, ".rds")

# --- LOAD EXPORTED SHAP DATA ---
shap_export_data <- readRDS(import_filename)

# Extract components
shap_values <- shap_export_data$shap_vals
feature_data <- shap_export_data$feature_data

# --- GENERATE PLOT ---
shp <- shapviz(shap_values, X = feature_data)


# --- Assuming 'shap_values' and 'x_test_matrix' are loaded/available here ---

# Generate the base shapviz plot
shp <- shapviz(shap_values, X = x_test_matrix[test_idx, ])

p_bee <- sv_importance(shp, kind = "beeswarm", max_display = nvari, show_numbers = TRUE) + 
  
  # Deploy the Elsevier-standard publicaton theme
  theme_pubr(base_size = 14) + 
  
  labs(
    title = paste("LSTM70 Spatial SHAP Distribution:", sim_name),
    x = "SHAP Value (Impact on Continuous TTI Prediction)",
    y = "Environmental Predictors"
  ) +
  
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 16, margin = margin(b = 15)),
    
    # Format the y-axis feature labels to be bold and clean
    axis.text.y = element_text(face = "bold", color = "black", size = 12),
    axis.text.x = element_text(color = "black", size = 11),
    axis.title = element_text(face = "bold", size = 13),
    
    # Retain a faint vertical grid line at 0.0 to anchor the SHAP visual spread
    panel.grid.major.x = element_line(color = "grey80", linetype = "dashed"),
    
    # Clean up the color bar legend
    legend.title = element_text(face = "bold", size = 12, angle = 90),
    legend.text = element_text(size = 10),
    legend.position = "right"
  )

print(p_bee)

# Save targeting exact Elsevier full-page specifications (190mm wide)
ggsave(filename = paste0("Figure3_LSTM70_Spatial_SHAP_Beeswarm_", sim_name, "_Elsevier.png"), 
       plot = p_bee, 
       width = 7.48, height = 6, dpi = 500, bg = "white")

message("Elsevier-compliant SHAP beeswarm plot successfully generated and saved.")

