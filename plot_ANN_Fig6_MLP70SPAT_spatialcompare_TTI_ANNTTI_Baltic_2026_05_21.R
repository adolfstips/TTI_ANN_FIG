#R


#2026.05.21

# Create spatial comparison plot 

# --- 1. SETUP & LIBRARIES ---
library(torch)
library(terra)
library(dplyr)
library(ggplot2)
library(tidyr)
library(scales) # Required for the color mapping rescale function
library(ggpubr) # Ensure ggpubr is loaded

device <- ifelse(cuda_is_available(), "cuda", "cpu")
torch_device(device)

model_name = "MLP70SPAT"
scenario="Comprehensive"
run_date="2026-05-26"

# --- 2. GRID CORRESPONDENCE & COORDINATE EXTRACTION ---
rfile <- './GES_Baltic_5year_ALLE_Combined_4D.nc'
sds_input <- terra::sds(rfile, guessCRS=TRUE)

Depth_rast <- sds_input$Depth[[1]]
ii.good <- which(!is.na(values(Depth_rast))) 
ngoodp <- length(ii.good)

coords <- terra::xyFromCell(Depth_rast, ii.good)
lon_lat_df <- data.frame(x = coords[, 1], y = coords[, 2])

target_years <- c(1970, 1995, 2020)
years_sequence <- seq(1955, 2020, by = 5)
target_indices <- match(target_years, years_sequence) 

vari <- c("Nit", "Pho", "Chla", "DO2_mg", "Secchi", "Temp", "Depth")
target_var <- "TTI"

# --- 3. MATRIX PREPARATION & MODEL LOADING ---
extracted_list <- list()
for (var_name in names(sds_input)) {
  extracted_list[[var_name]] <- as.vector(values(sds_input[[var_name]])[ii.good, ])
}
dataall <- as.data.frame(extracted_list)

for (p in vari) {
  dataall[[p]] <- scale(dataall[[p]])
}

cat("Loading saved model state: MLP70SPAT_Comprehensive_2026-05-26 ...\n")
model <- torch_load("./Models/MLP70SPAT_Comprehensive_2026-05-26.rt")
model$to(device = device)
model$eval()

x_tensor <- torch_tensor(as.matrix(dataall[, vari]), dtype = torch_float(), device = device)
with_no_grad({
  ann_tti_raw <- as.numeric(model(x_tensor)$cpu())
})
true_tti_raw <- dataall[[target_var]]

# --- 4. ASSEMBLE PANEL DATAFRAME ---
panel_collection <- data.frame()

for (i in seq_along(target_years)) {
  yr <- target_years[i]
  idx <- target_indices[i]
  
  start_row <- (idx - 1) * ngoodp + 1
  end_row   <- idx * ngoodp
  
  true_slice <- true_tti_raw[start_row:end_row]
  ann_slice  <- ann_tti_raw[start_row:end_row]
  
  df_true <- data.frame(
    Longitude = lon_lat_df$x,
    Latitude  = lon_lat_df$y,
    Year      = paste("Pentad Center:", yr),
    Type      = "Ecosystem Model\n(True TTI)", # Added \n here
    Value     = true_slice
  )
  
  df_ann <- data.frame(
    Longitude = lon_lat_df$x,
    Latitude  = lon_lat_df$y,
    Year      = paste("Pentad Center:", yr),
    Type      = "MLP70SPAT\n(Predicted TTI)", # Added \n here
    Value     = ann_slice
  )
  
  panel_collection <- rbind(panel_collection, df_true, df_ann)
}

# Update the factor levels so ggplot recognizes the new strings with the line breaks
panel_collection$Type <- factor(panel_collection$Type, 
                                levels = c("Ecosystem Model\n(True TTI)", "MLP70SPAT\n(Predicted TTI)"))

panel_collection$Year <- factor(panel_collection$Year, 
                                levels = paste("Pentad Center:", target_years))

# --- 5. HIGH-CONTRAST 3x2 SPECIFIC SELECTION GRID ---
library(ggpubr) # Ensure ggpubr is loaded

# --- 5. HIGH-CONTRAST 3x2 SPECIFIC SELECTION GRID (ELSEVIER OPTIMIZED) ---
p_spatial_grid <- ggplot(panel_collection, aes(x = Longitude, y = Latitude)) +
  geom_raster(aes(fill = Value)) +
  
  scale_fill_gradientn(
    colors = c("#d73027", "#f46d43", "#fdae61", "#fee08b", "#ffffbf", "#d9ef8b", "#a6d96a", "#1a9850"),
    values = scales::rescale(c(0.3, 0.5, 0.68, 0.70, 0.71, 0.72, 0.85, 1.0)),
    limits = c(0.4, 1.0),
    oob = scales::squish,
    name = "TTI Value:"
  ) +
  
  geom_contour(aes(z = Value), breaks = 0.71, colour = "black", linewidth = 0.6, alpha = 0.8) +
  
  # Geographic correction for the Baltic Sea
  coord_fixed(ratio = 1.7) +
  
  # Removing padding inside the facets to maximize map size
  scale_x_continuous(labels = function(x) paste0(x, "°E"), breaks = seq(10, 30, by = 10), expand = c(0, 0)) +
  scale_y_continuous(labels = function(y) paste0(y, "°N"), breaks = seq(54, 66, by = 4), expand = c(0, 0)) +
  
  facet_grid(Year ~ Type) +
  
  # Moved the subtitle information into the manuscript caption for a cleaner image
  labs(
    title = "Spatio-Temporal Reconstruction of Baltic Sea Trophic Status",
    x = "Longitude",
    y = "Latitude"
  ) +
  
  theme_pubr(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 16, margin = margin(b = 15)),
    
    # Strip (Facet) formatting
    strip.text = element_text(face = "bold", size = 12), # Reduced from 13 to 12
    strip.background = element_rect(fill = "grey95", color = "black", linewidth = 0.8),
    
    # Clean panel borders
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.8),
    panel.spacing = unit(1, "lines"), 
    
    # Legend relocated to the bottom to maximize horizontal map space
    legend.position = "bottom",
    legend.title = element_text(face = "bold", size = 14),
    legend.text = element_text(size = 12),
    legend.key.width = unit(2.5, "cm"), # Make it a wide, readable bar
    legend.margin = margin(t = 10),
    
    axis.text = element_text(color = "black", size = 11),
    axis.title = element_text(face = "bold", size = 13)
  )

print(p_spatial_grid)

# Save targeting exact Elsevier full-page specifications (190mm wide, taller to fit 3 rows)
ggsave("Figure6_Spatio_Temporal_TTI_Grid_Elsevier.png", plot = p_spatial_grid, 
       width = 7.48, height = 9.5, dpi = 500, bg = "white")

message("Elsevier-compliant spatio-temporal 3x2 grid saved successfully.")



