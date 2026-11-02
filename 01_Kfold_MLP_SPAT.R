# ==============================================================================
# Script: 01_Kfold_MLP_SPAT.R
# Title: 10-Fold Spatial Block Cross-Validation for Feedforward Network (MLP_SPAT)
# Reference: Stips et al. (2026) - Environmental Modelling & Software
# License: GPL-3.0
# ==============================================================================

library(torch)
library(terra)
library(dplyr)
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

model_name      <- "MLP_SPAT"
out_dir <- "./Output" 
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
rfile           <- "./GES_Baltic_5year_ALLE_Combined_4D.nc"

ntime           <- 14
n_splits        <- 10
block_size      <- 10       # 10x10 pixel blocks (40' Lon x 20' Lat)
target_var      <- "TTI"
tt_threshold    <- 0.71

batch_size      <- 2048
num_epochs      <- 150
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

# --- 2. BASE DATA LOADING & SPATIAL BLOCK GENERATION ---
sds_input  <- terra::sds(rfile, guessCRS = TRUE)
Depth_rast <- sds_input$Depth
ii.good    <- which(!is.na(values(Depth_rast[[1]]))) 
ngoodp     <- length(ii.good)

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

# --- 3. NON-OVERLAPPING 10-FOLD ASSIGNMENT ---
set.seed(123) 
unique_blocks    <- unique(dataall_raw$Block_ID)
n_blocks         <- length(unique_blocks)
fold_assignments <- sample(rep(1:n_splits, length.out = n_blocks))
block_fold_map   <- data.frame(Block_ID = unique_blocks, Fold_ID = fold_assignments)

dataall_raw <- left_join(dataall_raw, block_fold_map, by = "Block_ID")

# --- 4. 10-FOLD SPATIAL CROSS-VALIDATION LOOP ---
for (simul in seq_along(var_com)) {
  sim_name <- names(var_com)[simul]
  vari     <- var_com[[simul]]
  nvari    <- length(vari)
  
  cat(sprintf("\n\n==================================================\n"))
  cat(sprintf("Starting %d-Fold Spatial CV (%s) for: %s\n", n_splits, model_name, sim_name))
  cat(sprintf("==================================================\n"))
  
  for (k in 1:n_splits) {
    cat(sprintf("   Running Fold %d / %d...\n", k, n_splits))
    if (cuda_is_available()) { gc(); cuda_empty_cache() }
    
    # 4a. Strict 8-1-1 Fold Rotation (80% Train, 10% Val, 10% Test)
    val_k <- ifelse(k == n_splits, 1, k + 1) 
    
    dataTrain <- dataall_raw %>% filter(Fold_ID != k & Fold_ID != val_k)
    dataVal   <- dataall_raw %>% filter(Fold_ID == val_k)
    dataTest  <- dataall_raw %>% filter(Fold_ID == k)
    
    # 4b. Zero-Leakage Standardisation (fit strictly on Training Fold)
    for (p in vari) {
      mean_train <- mean(dataTrain[[p]], na.rm = TRUE)
      sd_train   <- sd(dataTrain[[p]], na.rm = TRUE)
      
      dataTrain[[p]] <- (dataTrain[[p]] - mean_train) / sd_train
      dataVal[[p]]   <- (dataVal[[p]] - mean_train) / sd_train
      dataTest[[p]]  <- (dataTest[[p]] - mean_train) / sd_train
    }
    
    # 4c. Tensor Preparation
    x_train <- torch_tensor(as.matrix(dataTrain[, vari]), dtype = torch_float(), device = device)
    y_train <- torch_tensor(as.matrix(dataTrain[[target_var]]), dtype = torch_float(), device = device)
    x_val   <- torch_tensor(as.matrix(dataVal[, vari]), dtype = torch_float(), device = device)
    y_val   <- torch_tensor(as.matrix(dataVal[[target_var]]), dtype = torch_float(), device = device)
    x_test  <- torch_tensor(as.matrix(dataTest[, vari]), dtype = torch_float(), device = device)
    y_test  <- torch_tensor(as.matrix(dataTest[[target_var]]), dtype = torch_float(), device = device)
    
    # 4d. MLP Architecture (128-64)
    model <- nn_sequential(
      nn_linear(nvari, 128), nn_relu(), nn_dropout(p = 0.2),
      nn_linear(128, 64),    nn_relu(), nn_dropout(p = 0.2),
      nn_linear(64, 1)
    )
    model$to(device = device)
    
    optimizer <- optim_adam(model$parameters, lr = learning_rate)
    criterion <- nn_mse_loss()
    
    # 4e. Training Loop with Dynamic Early Stopping
    n_train_rows  <- nrow(dataTrain)
    n_val_rows    <- nrow(dataVal)
    train_batches <- 1:((n_train_rows - 1) %/% batch_size + 1)
    val_batches   <- 1:((n_val_rows - 1) %/% batch_size + 1)
    
    best_loss             <- Inf
    best_model_state      <- NULL
    epochs_no_improvement <- 0
    
    for (epoch in 1:num_epochs) {
      model$train()
      shuffled_indices <- sample(1:n_train_rows)
      
      for (j in train_batches) {
        start_idx <- (j - 1) * batch_size + 1
        end_idx   <- min(j * batch_size, n_train_rows)
        idx       <- shuffled_indices[start_idx:end_idx]
        
        optimizer$zero_grad()
        loss <- criterion(model(x_train[idx, ]), y_train[idx]$view(c(-1, 1)))
        loss$backward()
        optimizer$step()
      }
      
      model$eval()
      val_loss <- 0
      with_no_grad({
        for (j in val_batches) {
          start_idx   <- (j - 1) * batch_size + 1
          end_idx     <- min(j * batch_size, n_val_rows)
          val_outputs <- model(x_val[start_idx:end_idx, ])
          v_loss      <- criterion(val_outputs, y_val[start_idx:end_idx]$view(c(-1, 1)))
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
        cat(sprintf("      Fold %d early stopped at epoch %d.\n", k, epoch))
        break
      }
    }
    
    if (!is.null(best_model_state)) {
      model$load_state_dict(best_model_state)
      model$to(device = device)
    }
    
    # 4f. Out-of-Sample Evaluation on Independent Test Fold
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
      Scenario = sim_name, Split = k, 
      RMSE = rmse_val, R2 = r2_val, ACC = acc, Precision = prec, Recall = rec, F1 = f1
    ))
    
    # 4g. Monte Carlo SHAP Extraction on Independent Test Fold
    predict_fun <- function(object, newdata) {
      batch_tensor <- torch_tensor(as.matrix(newdata), dtype = torch_float(), device = device)
      with_no_grad({ pred <- as.numeric(object(batch_tensor)$cpu()) })
      return(pred)
    }
    
    x_tr_mat <- as.matrix(dataTrain[, vari])
    x_te_mat <- as.matrix(dataTest[, vari])
    colnames(x_te_mat) <- vari
    
    set.seed(123 + k)
    bg_idx <- sample(nrow(x_tr_mat), min(500, nrow(x_tr_mat)))
    te_idx <- sample(nrow(x_te_mat), min(2000, nrow(x_te_mat)))
    
    shap_vals <- fastshap::explain(
      object = model, X = x_tr_mat[bg_idx, ], newdata = x_te_mat[te_idx, ],
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
        Scenario = sim_name, Split = k, Feature = feat, SHAP_Mean_Abs = as.numeric(global_imp[feat])
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

message("\nAll outputs successfully saved for ", model_name, ":\n  - ", summary_file,
        "\n  - ", raw_shap_file, "\n  - ", cf_file)
