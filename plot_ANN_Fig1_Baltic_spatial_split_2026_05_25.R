#R

#2026.05.24

# Plot the Baltic Sea spatial block splitting 
# --- 1. SETUP & LIBRARIES ---
library(terra)
library(dplyr)
library(ggplot2)
library(ggpubr) # Added for Elsevier-standard publication themes

# --- 2. DATA READING & TARGET ISOLATION ---
rfile <- './GES_Baltic_5year_ALLE_Combined_4D.nc'
sds_input <- terra::sds(rfile, guessCRS=TRUE)

Depth_rast <- sds_input$Depth[[1]] 
ii.good <- which(!is.na(values(Depth_rast))) 
ngoodp <- length(ii.good)

# --- 3. GENERATE 20x20 SPATIAL BLOCKS ---
rc <- terra::rowColFromCell(Depth_rast, ii.good)
block_size <- 20
block_row <- ceiling(rc[,1] / block_size)
block_col <- ceiling(rc[,2] / block_size)
spatial_blocks <- paste(block_row, block_col, sep="_")

# --- 4. TRAIN/VAL SPLIT (60:40) ---
set.seed(123)
unique_blocks <- unique(spatial_blocks)
train_blocks <- sample(unique_blocks, size = length(unique_blocks) * 0.60)

is_train <- ifelse(spatial_blocks %in% train_blocks, 1, 0)

# --- 5. PREPARE DATAFRAME FOR GGPLOT2 ---
block_map <- Depth_rast
terra::values(block_map) <- NA 
terra::values(block_map)[ii.good] <- is_train

map_df <- as.data.frame(block_map, xy = TRUE, na.rm = TRUE)
colnames(map_df)[3] <- "Split_Value"

map_df$Split <- factor(map_df$Split_Value, 
                       levels = c(1, 0), 
                       labels = c("Training Data (60%)", "Validation Data (40%)"))

# --- 6. PUBLICATION-READY GGPLOT2 (ELSEVIER OPTIMIZED) ---
p_map <- ggplot(data = map_df, aes(x = x, y = y, fill = Split)) +
  geom_raster() +
  scale_fill_manual(values = c("Training Data (60%)" = "#e66101", 
                               "Validation Data (40%)" = "#5e3c99")) + 
  
  # Adjusted ratio for true Baltic Sea geographic projection (~60 deg N)
  coord_fixed(ratio = 1.7) + 
  
  # scale_x/y expand=c(0,0) removes the empty white padding inside the plot frame
  #scale_x_continuous(labels = function(x) paste0(x, "°E"), expand = c(0, 0)) +
  #scale_y_continuous(labels = function(y) paste0(y, "°N"), expand = c(0, 0)) +
  scale_x_continuous(labels = function(x) paste0(x, "°E"), expand = c(0.02, 0.02)) +
  scale_y_continuous(labels = function(y) paste0(y, "°N"), expand = c(0.02, 0.02)) +
  labs(
    title = "Spatial Block Cross-Validation Design",
    subtitle = "Baltic Sea partitioned into 20x20 pixel geometric blocks",
    x = "Longitude",
    y = "Latitude",
    fill = "Data Allocation:"
  ) +
  
  # Deploying ggpubr theme for clean, solid axes
  theme_pubr(base_size = 14) + 
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 17),
    plot.subtitle = element_text(hjust = 0.5, size = 13, color = "grey30", margin = margin(b = 15)),
    legend.position = "bottom",
    legend.title = element_text(face = "bold", size = 13),
    legend.text = element_text(size = 12),
    
    # Adding a solid border box which maps require in many journals
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 1),
    
    # Optional: Keep a very faint grid to guide the eye across coordinates
    panel.grid.major = element_line(color = "grey92", linetype = "dashed"),
    
    axis.title = element_text(face = "bold", size = 14),
    axis.text = element_text(color = "black", size = 12)
  )

print(p_map)

# Save targeting exact Elsevier full-page specifications
# Width: 7.48 inches (190mm) @ 500 DPI = exactly 3740 pixels wide.
ggsave("Figure1_Spatial_Block_Split_Elsevier.png", plot = p_map, 
       width = 7.48, height = 9, dpi = 500, bg = "white")

message("Elsevier-compliant spatial block map successfully generated and saved.")



