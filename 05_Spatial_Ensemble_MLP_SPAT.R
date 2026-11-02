# ==============================================================================
# Script: 05_Spatial_Ensemble_MLP_SPAT.R
# Title: 10-Fold Spatial Bootstrap Ensemble Reconstruction (Figures 5, 6, & S3)
# Description: Trains the 10-fold spatial MLP_SPAT (Comprehensive) ensemble and
#              reconstructs full-basin TTI trajectories (1955-2020) and spatial
#              maps (1970, 1995, 2020) with pixel-wise uncertainty (SD).
# Reference: Stips et al. (2026) - Environmental Modelling & Software
# License: GPL-3.0
# ==============================================================================

library(torch)
library(terra)
library(dplyr)
library(Metrics)

# --- 1. SETUP & CONFIGURATION ---
set.seed(123)
torch_manual_seed(123)

device <- ifelse(cuda_is_available(), "cuda", "cpu")
torch_device(device)
if (cuda_is_available()) { gc(); cuda_empty_cache() }

model_name    <- "MLP_SPAT"
out_dir       <- "./Output"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

rfile         <- "./GES_Baltic_5year_ALLE_Combined_4D.nc"

ntime         <- 14
years         <- seq(1955, 2020, by = 5)
n_splits      <- 10
block_size    <- 10         # 10x10 pixel blocks (40' Lon x 20' Lat)
target_var    <- "TTI"
vari          <- c("Nit", "Pho", "Chla", "DO2_mg", "Secchi", "Temp", "Depth")
nvari         <- length(vari)

batch_size    <- 2048 
num_epochs    <- 150
patience      <- 20
learning_rate <- 0.002

# --- 2. DATA PREPARATION & SPATIAL COORDINATES ---
sds_input  <- terra::sds(rfile, guessCRS = TRUE)
Depth_rast <- sds_input$Depth
ii.good    <- which(!is.na(values(Depth_rast[[1]]))) 
ngoodp     <- length(ii.good)

# Extract geographic coordinates for spatial mapping and area-weighting
coords     <- terra::xyFromCell(Depth_rast[[1]], ii.good)
lon_lat_df <- data.frame(Longitude = coords[, 1], Latitude = coords[, 2])

extracted_list <- list()
for (var_name in names(sds_input)) {
  extracted_list[[var_name]] <- as.vector(values(sds_input[[var_name]])[ii.good, ])
}

# Log-transform Chlorophyll-a
extracted_list$Chla <- log10(extracted_list$Chla + 0.01)
dataall_raw <- as.data.frame(extracted_list)

# Generate 10x10 spatial block IDs
rc <- terra::rowColFromCell(Depth_rast[[1]], ii.good)
spatial_blocks <- paste(ceiling(rc[, 1] / block_size), ceiling(rc[, 2] / block_size), sep = "_")
dataall_raw$Block_ID <- rep(spatial_blocks, ntime)

# Non-overlapping 10-fold assignment (identical seed to 01_Kfold_MLP_SPAT.R)
set.seed(123)
unique_blocks    <- unique(dataall_raw$Block_ID)
n_blocks         <- length(unique_blocks)
fold_assignments <- sample(rep(1:n_splits, length.out = n_blocks))
block_fold_map   <- data.frame(Block_ID = unique_blocks, Fold_ID = fold_assignments)

dataall_raw <- left_join(dataall_raw, block_fold_map, by = "Block_ID")

# Matrix to store full-basin predictions across all 10 folds: (ngoodp * ntime) x 10
all_predictions_matrix <- matrix(NA, nrow = ngoodp * ntime, ncol = n_splits)

cat(sprintf("\n==================================================\n"))
cat(sprintf("Generating %d-Fold Spatial Ensemble (%s - Comprehensive)\n", n_splits, model_name))
cat(sprintf("==================================================\n"))

# --- 3. 10-FOLD ENSEMBLE TRAINING & FULL-BASIN RECONSTRUCTION LOOP ---
for (k in 1:n_splits) {
  cat(sprintf("   Training Ensemble Member %d / %d...\n", k, n_splits))
  if (cuda_is_available()) { gc(); cuda_empty_cache() }
  
  # 3a. Strict 8-1-1 Fold Rotation
  val_k <- ifelse(k == n_splits, 1, k + 1) 
  
  dataTrain      <- dataall_raw %>% filter(Fold_ID != k & Fold_ID != val_k)
  dataVal        <- dataall_raw %>% filter(Fold_ID == val_k)
  dataall_scaled <- dataall_raw 
  
  # 3b. Zero-Leakage Standardisation (fit strictly on Training Fold)
  for (p in vari) {
    mean_train <- mean(dataTrain[[p]], na.rm = TRUE)
    sd_train   <- sd(dataTrain[[p]], na.rm = TRUE)
    
    dataTrain[[p]]      <- (dataTrain[[p]] - mean_train) / sd_train
    dataVal[[p]]        <- (dataVal[[p]] - mean_train) / sd_train
    dataall_scaled[[p]] <- (dataall_scaled[[p]] - mean_train) / sd_train 
  }
  
  x_train <- torch_tensor(as.matrix(dataTrain[, vari]), dtype = torch_float(), device = device)
  y_train <- torch_tensor(as.matrix(dataTrain[[target_var]]), dtype = torch_float(), device = device)
  x_val   <- torch_tensor(as.matrix(dataVal[, vari]), dtype = torch_float(), device = device)
  y_val   <- torch_tensor(as.matrix(dataVal[[target_var]]), dtype = torch_float(), device = device)
  x_all   <- torch_tensor(as.matrix(dataall_scaled[, vari]), dtype = torch_float(), device = device)
  
  # 3c. MLP Architecture (128-64)
  model <- nn_sequential(
    nn_linear(nvari, 128), nn_relu(), nn_dropout(p = 0.2),
    nn_linear(128, 64),    nn_relu(), nn_dropout(p = 0.2),
    nn_linear(64, 1)
  )
  model$to(device = device)
  
  criterion <- nn_mse_loss()
  optimizer <- optim_adam(model$parameters, lr = learning_rate)
  
  n_train       <- nrow(dataTrain)
  n_val         <- nrow(dataVal)
  train_batches <- 1:((n_train - 1) %/% batch_size + 1)
  val_batches   <- 1:((n_val - 1) %/% batch_size + 1)
  
  best_loss             <- Inf
  best_model_state      <- NULL
  epochs_no_improvement <- 0
  
  # 3d. Training Loop with Dynamic Early Stopping
  for (epoch in 1:num_epochs) {
    model$train()
    shuffled_indices <- sample(1:n_train)
    for (j in train_batches) {
      start_index <- (j - 1) * batch_size + 1
      end_index   <- min(j * batch_size, n_train)
      idx         <- shuffled_indices[start_index:end_index]
      
      optimizer$zero_grad()
      loss <- criterion(model(x_train[idx, ]), y_train[idx]$view(c(-1, 1)))
      loss$backward()
      optimizer$step()
    }
    
    model$eval()
    val_loss <- 0
    with_no_grad({
      for (j in val_batches) {
        start_index <- (j - 1) * batch_size + 1
        end_index   <- min(j * batch_size, n_val)
        v_loss      <- criterion(model(x_val[start_index:end_index, ]), y_val[start_index:end_index]$view(c(-1, 1)))
        val_loss    <- val_loss + v_loss$item()
      }
    })
    
    avg_val_loss <- val_loss / length(val_batches)
    if (avg_val_loss < best_loss) {
      best_loss             <- avg_val_loss
      epochs_no_improvement <- 0
      best_model_state      <- lapply(model$state_dict(), function(x) x$cpu()$clone())
    } else {
      epochs_no_improvement <- epochs_no_improvement + 1
    }
    if (epochs_no_improvement >= patience) {
      cat(sprintf("      Member %d early stopped at epoch %d.\n", k, epoch))
      break
    }
  }
  
  if (!is.null(best_model_state)) {
    model$load_state_dict(best_model_state)
    model$to(device = device)
  }
  model$eval()
  
  # 3e. Predict full basin for Member k
  with_no_grad({ 
    all_predictions_matrix[, k] <- as.numeric(model(x_all)$cpu()) 
  })
}

# --- 4. PIXEL-WISE & BASIN-WIDE FISHER-Z ENSEMBLE STATISTICS ---
cat("\nCalculating Fisher-Z ensemble statistics for Figures 5, 6, and S3...\n")

# Transform predictions to Fisher Z-space (bounded to prevent Inf at +-1)
z_matrix <- atanh(pmin(pmax(all_predictions_matrix, -0.999), 0.999))

# 4a. Pixel-wise Mean and Standard Deviation across the 10 folds
pixel_z_mean   <- rowMeans(z_matrix)
pixel_z_sd     <- apply(z_matrix, 1, sd)
pixel_tti_mean <- tanh(pixel_z_mean)
pixel_tti_sd   <- tanh(pixel_z_sd)

final_map_df <- data.frame(
  Longitude         = rep(lon_lat_df$Longitude, ntime),
  Latitude          = rep(lon_lat_df$Latitude, ntime),
  Year              = rep(years, each = ngoodp),
  True_TTI          = dataall_raw[[target_var]],
  Ensemble_Mean_TTI = pixel_tti_mean,
  Ensemble_SD_TTI   = pixel_tti_sd
)

# 4b. Export Spatial Map Data for Figure 6 & Figure S3 (1970, 1995, 2020)
target_years <- c(1970, 1995, 2020)
fig6_data    <- final_map_df %>% filter(Year %in% target_years)
map_file     <- file.path(out_dir, paste0("Ensemble_Map_Data_", model_name, ".rds"))
saveRDS(fig6_data, file = map_file)

# 4c. Export Area-Weighted Basin-Wide Time Series (1955-2020) for Figure 5
cos_weights <- cos(rep(lon_lat_df$Latitude, ntime) * pi / 180)
true_z      <- atanh(pmin(pmax(dataall_raw[[target_var]], -0.999), 0.999))

fold_basin_means <- matrix(NA, nrow = ntime, ncol = n_splits)
true_basin_mean  <- numeric(ntime)
true_basin_sd    <- numeric(ntime)

for (t_idx in 1:ntime) {
  rows_t <- ((t_idx - 1) * ngoodp + 1):(t_idx * ngoodp)
  w_t    <- cos_weights[rows_t] / sum(cos_weights[rows_t])
  
  true_basin_mean[t_idx] <- tanh(sum(w_t * true_z[rows_t]))
  true_basin_sd[t_idx]   <- sd(dataall_raw[[target_var]][rows_t], na.rm = TRUE)
  
  for (k in 1:n_splits) {
    fold_basin_means[t_idx, k] <- tanh(sum(w_t * z_matrix[rows_t, k]))
  }
}

fig5_timeseries_df <- data.frame(
  Year          = years,
  True_TTI_Mean = true_basin_mean,
  True_TTI_SD   = true_basin_sd,
  Pred_TTI_Mean = rowMeans(fold_basin_means),
  Pred_TTI_CI   = 1.96 * apply(fold_basin_means, 1, sd) / sqrt(n_splits)
)

ts_file <- file.path(out_dir, paste0("Ensemble_TimeSeries_", model_name, ".rds"))
saveRDS(fig5_timeseries_df, file = ts_file)

message("\nSpatial ensemble outputs successfully saved:\n  - ", map_file, "\n  - ", ts_file)

