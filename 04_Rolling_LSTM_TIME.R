# ==============================================================================
# Script: 04_Rolling_LSTM_TIME.R
# Title: Rolling-Origin Temporal Forecast for Recurrent Network (LSTM_TIME)
# Reference: Stips et al. (2026) - Environmental Modelling & Software
# License: GPL-3.0
# ==============================================================================

library(torch)
library(terra)
library(dplyr)
library(abind)
library(fastshap)
library(Metrics)
library(cvms)
library(tibble)

# --- 1. SETUP & CONFIGURATION ---
set.seed(123)
torch_manual_seed(123)

device <- ifelse(cuda_is_available(), "cuda", "cpu")
torch_device(device)
if (cuda_is_available()) { gc(); cuda_empty_cache() }

model_name      <- "LSTM_TIME"
out_dir <- "./Output" 
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
rfile           <- "./GES_Baltic_5year_ALLE_Combined_4D.nc"

ntime           <- 14  
sequence_length <- 7        # 7 pentads = 35-year chronological memory window
block_size      <- 20
target_var      <- "TTI"
tt_threshold    <- 0.71  

batch_size      <- 2048
num_epochs      <- 250
patience        <- 20
learning_rate   <- 0.002

var_com <- list(
  MSFD_Baseline  = c("Nit", "Pho", "Chla", "DO2_mg", "Secchi"), 
  Remote_Sensing = c("Temp", "Chla", "Depth", "Secchi"), 
  Comprehensive  = c("Nit", "Pho", "Chla", "DO2_mg", "Secchi", "Temp", "Depth")
)

metrics_collection <- data.frame()
shap_collection    <- data.frame()

raw_shap_export <- list(
  Comprehensive  = list(shap_vals = NULL, feature_data = NULL),
  MSFD_Baseline  = list(shap_vals = NULL, feature_data = NULL),
  Remote_Sensing = list(shap_vals = NULL, feature_data = NULL)
)

global_targets <- c()
global_preds   <- c()

# Sequence-to-One extraction module (explicitly selects the final time step)
extract_last_step <- nn_module(
  "extract_last_step",
  initialize = function() {},
  forward = function(x) { 
    out <- x[[1]]
    return(out[, sequence_length, ]) 
  }
)

# --- 2. GRID EXTRACTION & 3D RESTRUCTURING ---
sds_input  <- terra::sds(rfile, guessCRS = TRUE)
Depth_rast <- sds_input$Depth
ii.good    <- which(!is.na(values(Depth_rast[[1]]))) 
ngoodp     <- length(ii.good)  

rc <- terra::rowColFromCell(Depth_rast[[1]], ii.good)
spatial_blocks <- paste(ceiling(rc[, 1] / block_size), ceiling(rc[, 2] / block_size), sep = "_")

var_names <- names(sds_input)
num_vars  <- length(var_names)

master_3d <- array(NA, dim = c(ngoodp, ntime, num_vars))
dimnames(master_3d) <- list(NULL, NULL, var_names)

for (v in seq_along(var_names)) {
  master_3d[, , v] <- as.matrix(values(sds_input[[var_names[v]]])[ii.good, ])
}

# Log-transform Chlorophyll-a
master_3d[, , "Chla"] <- log10(master_3d[, , "Chla"] + 0.01)

# --- 3. SPATIAL PARTITIONING & 3D SEQUENCE GENERATOR ---
set.seed(123)
unique_blocks <- unique(spatial_blocks)
n_blocks      <- length(unique_blocks)

train_size <- floor(n_blocks * 0.60)
val_size   <- floor(n_blocks * 0.20)

train_blocks <- sample(unique_blocks, size = train_size)
rem_blocks   <- setdiff(unique_blocks, train_blocks)
val_blocks   <- sample(rem_blocks, size = val_size)
test_blocks  <- setdiff(rem_blocks, val_blocks)

train_idx <- which(spatial_blocks %in% train_blocks)
val_idx   <- which(spatial_blocks %in% val_blocks)
test_idx  <- which(spatial_blocks %in% test_blocks)

create_sequences <- function(data_array, node_indices, target_time_steps, features, target) {
  x_list <- list()
  y_list <- list()
  for (t in target_time_steps) {
    if (t < sequence_length) next 
    seq_x <- data_array[node_indices, (t - sequence_length + 1):t, features, drop = FALSE]
    seq_y <- data_array[node_indices, t, target]
    x_list[[length(x_list) + 1]] <- seq_x
    y_list[[length(y_list) + 1]] <- seq_y
  }
  X <- do.call(abind::abind, c(x_list, along = 1))
  Y <- do.call(c, y_list)
  return(list(X = X, Y = Y))
}

# --- 4. ROLLING-ORIGIN TEMPORAL FORECAST LOOP (1995-2020) ---
for (simul in seq_along(var_com)) {
  sim_name <- names(var_com)[simul]
  vari     <- var_com[[simul]]
  nvari    <- length(vari)
  
  cat(sprintf("\n\n==================================================\n"))
  cat(sprintf("Executing Rolling Forecast (%s) for: %s\n", model_name, sim_name))
  cat(sprintf("==================================================\n"))
  
  for (t_train_end in 7:12) {
    t_val         <- t_train_end + 1
    t_test        <- t_train_end + 2
    forecast_year <- 1950 + (t_test * 5)
    
    cat(sprintf("\n   --- Rolling Origin: Train %d-%d | Val %d | Forecast %d (%d) ---\n", 
                sequence_length, t_train_end, t_val, t_test, forecast_year))
    
    if (cuda_is_available()) { gc(); cuda_empty_cache() }
    
    # 4a. Dynamic Zero-Leakage Standardisation (fit strictly on historical training window)
    fold_3d <- master_3d 
    for (p in vari) {
      mean_train     <- mean(fold_3d[train_idx, 1:t_train_end, p], na.rm = TRUE)
      sd_train       <- sd(fold_3d[train_idx, 1:t_train_end, p], na.rm = TRUE)
      fold_3d[, , p] <- (fold_3d[, , p] - mean_train) / sd_train
    }
    
    # 4b. Prepare 3D Sequence Datasets
    train_data <- create_sequences(fold_3d, train_idx, sequence_length:t_train_end, vari, target_var)
    val_data   <- create_sequences(fold_3d, val_idx, t_val, vari, target_var)
    test_data  <- create_sequences(fold_3d, test_idx, t_test, vari, target_var)
    
    rm(fold_3d); gc()
    
    x_train <- torch_tensor(train_data$X, dtype = torch_float(), device = device)
    y_train <- torch_tensor(train_data$Y, dtype = torch_float(), device = device)
    x_val   <- torch_tensor(val_data$X, dtype = torch_float(), device = device)
    y_val   <- torch_tensor(val_data$Y, dtype = torch_float(), device = device)
    x_test  <- torch_tensor(test_data$X, dtype = torch_float(), device = device)
    y_test  <- torch_tensor(test_data$Y, dtype = torch_float(), device = device)
    
    # 4c. Stacked LSTM Architecture (128-64, Sequence-to-One)
    model <- nn_sequential(
      nn_lstm(input_size = nvari, hidden_size = 128, batch_first = TRUE),
      nn_module("extract_first_lstm", forward = function(x) { return(x[[1]]) })(),
      nn_dropout(p = 0.2),
      nn_lstm(input_size = 128, hidden_size = 64, batch_first = TRUE),
      extract_last_step(),
      nn_dropout(p = 0.2),
      nn_linear(64, 1)
    )
    model$to(device = device)
    
    criterion <- nn_mse_loss()
    optimizer <- optim_adam(model$parameters, lr = learning_rate)
    
    n_train_rows  <- dim(x_train)[1]
    n_val_rows    <- dim(x_val)[1]
    train_batches <- 1:((n_train_rows - 1) %/% batch_size + 1)
    val_batches   <- 1:((n_val_rows - 1) %/% batch_size + 1)
    
    best_loss             <- Inf
    best_model_state      <- NULL
    epochs_no_improvement <- 0
    
    # 4d. Training Loop with Dynamic Early Stopping
    for (epoch in 1:num_epochs) {
      model$train()
      shuffled_indices <- sample(1:n_train_rows)
      
      for (j in train_batches) {
        start_idx <- (j - 1) * batch_size + 1
        end_idx   <- min(j * batch_size, n_train_rows)
        idx       <- shuffled_indices[start_idx:end_idx]
        
        optimizer$zero_grad()
        loss <- criterion(model(x_train[idx, , ]), y_train[idx]$view(c(-1, 1)))
        loss$backward()
        optimizer$step()
      }
      
      model$eval()
      val_loss <- 0
      with_no_grad({
        for (j in val_batches) {
          start_idx <- (j - 1) * batch_size + 1
          end_idx   <- min(j * batch_size, n_val_rows)
          v_loss    <- criterion(model(x_val[start_idx:end_idx, , ]), y_val[start_idx:end_idx]$view(c(-1, 1)))
          val_loss  <- val_loss + v_loss$item()
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
      if (epochs_no_improvement >= patience) break
    }
    
    if (!is.null(best_model_state)) {
      model$load_state_dict(best_model_state)
      model$to(device = device)
    }
    
    # 4e. Out-of-Sample Evaluation on Unseen Target Pentad
    model$eval()
    with_no_grad({ prob_preds <- as.numeric(model(x_test)$cpu()) })
    true_tti <- as.numeric(y_test$cpu())
    
    pred_ges <- ifelse(prob_preds > tt_threshold, 1, 0)
    true_ges <- ifelse(true_tti > tt_threshold, 1, 0)
    
    if (sim_name == "Comprehensive") {
      global_targets <- c(global_targets, true_ges)
      global_preds   <- c(global_preds, pred_ges)
    }
    
    TP <- sum(pred_ges == 1 & true_ges == 1)
    FP <- sum(pred_ges == 1 & true_ges == 0)
    FN <- sum(pred_ges == 0 & true_ges == 1)
    TN <- sum(pred_ges == 0 & true_ges == 0)
    
    acc      <- (TP + TN) / (TP + TN + FP + FN)
    prec     <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
    rec      <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
    f1       <- ifelse((prec + rec) == 0, 0, 2 * (prec * rec) / (prec + rec))
    mse_raw  <- Metrics::mse(true_tti, prob_preds)
    rmse_val <- sqrt(mse_raw)
    r2_val   <- 1 - (sum((true_tti - prob_preds)^2) / sum((true_tti - mean(true_tti))^2))
    
    metrics_collection <- rbind(metrics_collection, data.frame(
      Scenario = sim_name, Origin_Year = forecast_year,
      RMSE = rmse_val, R2 = r2_val, ACC = acc, Precision = prec, Recall = rec, F1 = f1,
      Train_Blocks = train_size, Test_Blocks = length(test_blocks)
    ))
    
    # 4f. Monte Carlo SHAP Extraction (2D-to-3D Sequence Wrapper for fastshap)
    predict_fun <- function(object, newdata) {
      n_samples <- nrow(newdata)
      seq_data  <- array(newdata, dim = c(n_samples, length(vari), sequence_length))
      seq_data  <- aperm(seq_data, c(1, 3, 2)) 
      batch_tensor <- torch_tensor(seq_data, dtype = torch_float(), device = device)
      with_no_grad({ pred <- as.numeric(object(batch_tensor)$cpu()) })
      return(pred)
    }
    
    x_tr_3d <- as.array(train_data$X)
    x_te_3d <- as.array(test_data$X)
    
    set.seed(123)
    bg_idx <- sample(dim(x_tr_3d)[1], min(500, dim(x_tr_3d)[1]))
    te_idx <- sample(dim(x_te_3d)[1], min(2000, dim(x_te_3d)[1]))
    
    X_bg_2d   <- x_tr_3d[bg_idx, sequence_length, ]
    X_test_2d <- x_te_3d[te_idx, sequence_length, ]
    colnames(X_test_2d) <- vari
    
    shap_vals <- fastshap::explain(
      object = model, X = X_bg_2d, newdata = X_test_2d,
      nsim = 50, pred_wrapper = predict_fun, shap_only = FALSE
    )
    
    raw_shap_export[[sim_name]]$shap_vals <- rbind(
      raw_shap_export[[sim_name]]$shap_vals, shap_vals$shapley_values
    )
    raw_shap_export[[sim_name]]$feature_data <- rbind(
      raw_shap_export[[sim_name]]$feature_data, shap_vals$feature_values
    )
    
    global_imp <- colMeans(abs(as.matrix(shap_vals$shapley_values)))
    for (feat in names(global_imp)) {
      shap_collection <- rbind(shap_collection, data.frame(
        Scenario = sim_name, Origin_Year = forecast_year, Feature = feat, SHAP_Mean_Abs = as.numeric(global_imp[feat])
      ))
    }
  }
}

# --- 5. STATISTICAL AGGREGATION & HARMONIZED EXPORT ---
final_metrics <- metrics_collection %>%
  group_by(Scenario) %>%
  summarize(
    RMSE_Mean = mean(RMSE, na.rm = TRUE),      RMSE_CI = 1.96 * (sd(RMSE, na.rm = TRUE) / sqrt(n())),
    R2_Mean   = mean(R2, na.rm = TRUE),        R2_CI   = 1.96 * (sd(R2, na.rm = TRUE) / sqrt(n())),
    ACC_Mean  = mean(ACC, na.rm = TRUE),       ACC_CI  = 1.96 * (sd(ACC, na.rm = TRUE) / sqrt(n())),
    Prec_Mean = mean(Precision, na.rm = TRUE), Prec_CI = 1.96 * (sd(Precision, na.rm = TRUE) / sqrt(n())),
    Rec_Mean  = mean(Recall, na.rm = TRUE),    Rec_CI  = 1.96 * (sd(Recall, na.rm = TRUE) / sqrt(n())),
    F1_Mean   = mean(F1, na.rm = TRUE),        F1_CI   = 1.96 * (sd(F1, na.rm = TRUE) / sqrt(n())),
    N_Splits  = n(), 
    .groups   = "drop"
  )
print(final_metrics)

final_shap <- shap_collection %>%
  group_by(Scenario, Feature) %>%
  summarize(
    SHAP_Mean = mean(SHAP_Mean_Abs, na.rm = TRUE), 
    SHAP_CI   = 1.96 * (sd(SHAP_Mean_Abs, na.rm = TRUE) / sqrt(n())),
    .groups   = "drop"
  ) %>%
  arrange(Scenario, desc(SHAP_Mean)) 
print(final_shap)

# 5a. Save Summary Metrics & SHAP
summary_export_data <- list(
  metrics     = final_metrics,
  raw_metrics = metrics_collection, 
  final_shap  = final_shap,
  raw_shap    = shap_collection     
)
summary_file <- file.path(out_dir, paste0("Summary_", model_name, ".rds"))
saveRDS(summary_export_data, file = summary_file)

# 5b. Save Raw Point-Level SHAP
raw_shap_file <- file.path(out_dir, paste0("SHAP_RAW_", model_name, ".rds"))
saveRDS(raw_shap_export, file = raw_shap_file)

# 5c. Save Global Confusion Matrix (Comprehensive Scenario)
data_cm   <- tibble("target" = global_targets, "prediction" = global_preds)
eval_cm   <- cvms::evaluate(data = data_cm, target_col = "target", prediction_cols = "prediction", type = "binomial")
cf_matrix <- eval_cm$`Confusion Matrix`[[1]]
cf_file   <- file.path(out_dir, paste0("CF_Matrix_", model_name, ".rds"))
saveRDS(cf_matrix, file = cf_file)

message("\nAll outputs successfully saved for ", model_name, ":\n  - ", summary_file, "\n  - ", 
        raw_shap_file, "\n  - ", cf_file)
