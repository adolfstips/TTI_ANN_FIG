# ==============================================================================
# Script: 07_Plot_Fig1_Spatial_Block_Map.R
# Title: Generate Figure 1 (Spatial Block Cross-Validation Design Map)
# Description: Maps the exact 10x10 pixel block allocation for Fold 1 (80% Train,
#              10% Validation, 10% Testing) with geographic context.
# Reference: Stips et al. (2026) - Environmental Modelling & Software
# License: GPL-3.0
# ==============================================================================

library(terra)
library(dplyr)
library(ggplot2)
library(ggpubr)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)

# --- 1. DIRECTORY & DATA SETUP ---
fig_dir <- "./Figures"
if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)

rfile <- "./GES_Baltic_5year_ALLE_Combined_4D.nc"
if (!file.exists(rfile)) stop("Input NetCDF file not found: ", rfile)

sds_input  <- terra::sds(rfile, guessCRS = TRUE)
Depth_rast <- sds_input$Depth[[1]] 
ii.good    <- which(!is.na(values(Depth_rast))) 

# --- 2. GENERATE 10x10 SPATIAL BLOCKS ---
block_size     <- 10
rc             <- terra::rowColFromCell(Depth_rast, ii.good)
spatial_blocks <- paste(ceiling(rc[, 1] / block_size), ceiling(rc[, 2] / block_size), sep = "_")

# --- 3. EXACT 10-FOLD ASSIGNMENT (FOLD 1 ALLOCATION: 8-1-1 SPLIT) ---
# Uses the identical seed and assignment logic as 01_Kfold_MLP_SPAT.R & 02_Kfold_LSTM_SPAT.R
set.seed(123)
unique_blocks    <- unique(spatial_blocks)
n_blocks         <- length(unique_blocks)
fold_assignments <- sample(rep(1:10, length.out = n_blocks))
block_fold_map   <- data.frame(Block_ID = unique_blocks, Fold_ID = fold_assignments)

k     <- 1  # Representative Test Fold 1
val_k <- 2  # Representative Validation Fold 2

pixel_folds <- block_fold_map$Fold_ID[match(spatial_blocks, block_fold_map$Block_ID)]

split_assignment <- rep(1, length(spatial_blocks)) # Default: Training (Folds 3-10; 80%)
split_assignment[pixel_folds == val_k] <- 2        # Validation (Fold 2; 10%)
split_assignment[pixel_folds == k]     <- 3        # Testing (Fold 1; 10%)

block_map <- Depth_rast
terra::values(block_map)          <- NA 
terra::values(block_map)[ii.good] <- split_assignment

map_df <- as.data.frame(block_map, xy = TRUE, na.rm = TRUE)
colnames(map_df)[3] <- "Split_Value"

map_df$Split <- factor(
  map_df$Split_Value, 
  levels = c(1, 2, 3), 
  labels = c("Training Blocks (80%)", "Validation Blocks (10%)", "Testing Blocks (10%)")
)

# --- 4. LOAD GEOGRAPHIC FEATURES (Natural Earth) ---
baltic_countries <- ne_countries(scale = 10, returnclass = "sf", continent = "Europe")
rivers <- ne_download(scale = 10, type = "rivers_lake_centerlines", category = "physical", returnclass = "sf")

# --- 5. GENERATE PUBLICATION PLOT (FIGURE 1) ---
cb_palette <- c(
  "Training Blocks (80%)"   = "#E69F00",
  "Validation Blocks (10%)" = "#56B4E9",
  "Testing Blocks (10%)"    = "#009E73"
)

p_fig1 <- ggplot() +
  geom_raster(data = map_df, aes(x = x, y = y, fill = Split)) +
  scale_fill_manual(values = cb_palette) + 
  geom_sf(data = baltic_countries, fill = NA, color = "black", linewidth = 0.4) +
  geom_sf(data = rivers, color = "#1f78b4", linewidth = 0.2) +
  coord_sf(xlim = c(9.5, 30.5), ylim = c(53.5, 66.0), expand = FALSE) +
  scale_x_continuous(breaks = seq(10, 30, 5), labels = function(x) paste0(x, "\u00B0E")) +
  scale_y_continuous(breaks = seq(54, 66, 2), labels = function(y) paste0(y, "\u00B0N")) +
  labs(
    title    = "Spatial Block Cross-Validation Design",
    subtitle = "Baltic Sea partitioned into 10x10 pixel blocks (40' Lon \u00D7 20' Lat; ~31-43 km)",
    x        = "Longitude",
    y        = "Latitude",
    fill     = "10-Fold Split:"
  ) +
  guides(fill = guide_legend(nrow = 1, byrow = TRUE)) +
  theme_pubr(base_size = 14) + 
  theme(
    plot.title       = element_text(face = "bold", hjust = 0.5, size = 17),
    plot.subtitle    = element_text(hjust = 0.5, size = 13, color = "grey30", margin = margin(b = 15)),
    legend.position  = "bottom",
    legend.title     = element_text(face = "bold", size = 13),
    legend.text      = element_text(size = 12),
    legend.margin    = margin(t = -5),
    panel.border     = element_rect(colour = "black", fill = NA, linewidth = 1.2),
    panel.grid.major = element_line(color = "grey85", linetype = "dashed"),
    panel.background = element_rect(fill = "#f0f8ff"), 
    axis.title       = element_text(face = "bold", size = 14),
    axis.text        = element_text(color = "black", size = 12)
  )

out_file <- file.path(fig_dir, "Figure1_Spatial_Block_Geographic.png")
ggsave(out_file, plot = p_fig1, width = 8.5, height = 9, dpi = 500, bg = "white")
message("Figure 1 successfully saved to: ", out_file)

