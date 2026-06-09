#R

#2026.05.26

# Simplified K-fold cross validation LSTM70SPAT (Corrected Dimensions)

## This takes a very long time (hours)!

library(torch)
library(terra)
library(dplyr)
library(fastshap)
library(Metrics)


device <- ifelse(cuda_is_available(), "cuda", "cpu")
torch_device(device)

if (cuda_is_available()) {
  gc(); cuda_empty_cache() 
}

model_name = "LSTM70SPAT"
ntime <- 14
target_var <- "TTI"
tt_threshold <- 0.71
trainr <- 0.6

# Register the selection layer for LSTM output
select_lstm_output <- nn_module(
  "select_lstm_output",
  initialize = function() {},
  forward = function(x) { return(x[[1]]) }
)

# Variable Scenarios
var_com = list(
  MSFD_Baseline  = c("Nit", "Pho", "Chla", "DO2_mg", "Secchi"), 
  Remote_Sensing = c("Temp", "Chla", "Depth", "Secchi"), 
  Comprehensive  = c("Nit", "Pho", "Chla", "DO2_mg", "Secchi", "Temp", "Depth")
)

n_splits <- 10 

metrics_collection <- data.frame()
shap_collection <- data.frame()

# --- 2. BASE DATA LOADING ---
rfile <- './GES_Baltic_5year_ALLE_Combined_4D.nc'
sds_input <- terra::sds(rfile, guessCRS=TRUE)
Depth_rast <- sds_input$Depth
ii.good <- which(!is.na(values(Depth_rast[[1]]))) 
ngoodp <- length(ii.good)

extracted_list <- list()
for (var_name in names(sds_input)) {
  extracted_list[[var_name]] <- as.vector(values(sds_input[[var_name]])[ii.good, ])
}
dataall_raw <- as.data.frame(extracted_list)

rc <- terra::rowColFromCell(Depth_rast[[1]], ii.good)
spatial_blocks <- paste(ceiling(rc[,1] / 20), ceiling(rc[,2] / 20), sep="_")
dataall_raw$Block_ID <- rep(spatial_blocks, ntime)

# --- 3. THE K-FOLD SPATIAL LOOP ---
for (simul in seq_along(var_com)) {
  sim_name <- names(var_com)[simul]
  vari <- var_com[[simul]]
  nvari <- length(vari)
  
  cat(sprintf("\n\n>>> Starting %d-Fold Spatial Robustness for: %s <<<\n", n_splits, sim_name))
  
  for (k in 1:n_splits) {
    cat(sprintf("   Running Split %d / %d...\n", k, n_splits))
    
    if (cuda_is_available()) {
      gc(); cuda_empty_cache()
    }
    
    set.seed(123 + k) 
    dataall <- dataall_raw
    
    for (p in vari) dataall[[p]] <- scale(dataall[[p]]) 
    
    unique_blocks <- unique(dataall$Block_ID)
    train_blocks <- sample(unique_blocks, size = length(unique_blocks) * 0.60)
    
    dataTrain <- dataall %>% filter(Block_ID %in% train_blocks)
    dataTest  <- dataall %>% filter(!(Block_ID %in% train_blocks))
    
    # 3b. Tensor Prep (CRITICAL FIX: $unsqueeze(2) creates [Batch, 1, Features])
    x_train <- torch_tensor(as.matrix(dataTrain[, vari]), dtype=torch_float(), device=device)$unsqueeze(2)
    y_train <- torch_tensor(as.matrix(dataTrain[[target_var]]), dtype=torch_float(), device=device)
    x_test  <- torch_tensor(as.matrix(dataTest[, vari]), dtype=torch_float(), device=device)$unsqueeze(2)
    y_test  <- torch_tensor(as.matrix(dataTest[[target_var]]), dtype=torch_float(), device=device)
    
    # LSTM Architecture
    model <- nn_sequential(
      nn_lstm(input_size = nvari, hidden_size = 128, batch_first = TRUE),
      select_lstm_output(),
      nn_dropout(p = 0.2),
      
      nn_lstm(input_size = 128, hidden_size = 64, batch_first = TRUE),
      select_lstm_output(),
      nn_dropout(p = 0.2),
      
      nn_flatten(start_dim = 2, end_dim = 3), 
      nn_linear(64, 1)
    )
    
    model$to(device=device)
    
    criterion <- nn_mse_loss()
    optimizer <- optim_adam(model$parameters, lr = 0.002)
    
    # 3d. Training Loop
    batch_size <- 1024
    n_train_rows <- nrow(dataTrain)
    n_val_rows   <- nrow(dataTest)
    
    train_batches <- 1:((n_train_rows - 1) %/% batch_size + 1)
    val_batches   <- 1:((n_val_rows - 1) %/% batch_size + 1)
    
    best_loss <- Inf
    best_model_state <- NULL
    epochs_no_improvement <- 0
    patience <- 10
    
    for (epoch in 1:150) {
      model$train()
      shuffled_indices <- sample(1:n_train_rows)
      
      for (j in train_batches) {
        start_idx <- (j-1)*batch_size + 1
        end_idx   <- min(j*batch_size, n_train_rows)
        idx <- shuffled_indices[start_idx:end_idx]
        
        optimizer$zero_grad()
        # CRITICAL FIX: 3D Slicing [idx, , ]
        loss <- criterion(model(x_train[idx, , ]), y_train[idx]$view(c(-1, 1)))
        loss$backward()
        optimizer$step()
      }
      
      model$eval()
      val_loss <- 0
      with_no_grad({
        for (j in val_batches) {
          start_idx <- (j-1)*batch_size + 1
          end_idx   <- min(j*batch_size, n_val_rows)
          
          # CRITICAL FIX: 3D Slicing [start_idx:end_idx, , ]
          val_outputs <- model(x_test[start_idx:end_idx, , ])
          v_loss <- criterion(val_outputs, y_test[start_idx:end_idx]$view(c(-1, 1)))
          val_loss <- val_loss + v_loss$item()
        }
      })
      
      avg_val_loss <- val_loss / length(val_batches)
      
      if (avg_val_loss < best_loss) {
        best_loss <- avg_val_loss
        epochs_no_improvement <- 0
        best_model_state <- lapply(model$state_dict(), function(x) x$cpu()$clone())
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
      model$to(device=device)
    }
    
    # --- 3e. Evaluation ---
    model$eval()
    with_no_grad({ prob_preds <- as.numeric(model(x_test)$cpu()) })
    true_tti <- as.numeric(y_test$cpu())
    
    pred_ges <- ifelse(prob_preds > tt_threshold, 1, 0)
    true_ges <- ifelse(true_tti > tt_threshold, 1, 0)
    
    TP <- sum(pred_ges == 1 & true_ges == 1)
    FP <- sum(pred_ges == 1 & true_ges == 0)
    FN <- sum(pred_ges == 0 & true_ges == 1)
    TN <- sum(pred_ges == 0 & true_ges == 0)
    
    acc <- (TP + TN) / (TP + TN + FP + FN)
    prec <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
    rec  <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
    f1   <- ifelse((prec + rec) == 0, 0, 2 * (prec * rec) / (prec + rec))
    mse_val <- Metrics::mse(true_tti, prob_preds)
    
    metrics_collection <- rbind(metrics_collection, data.frame(
      Scenario = sim_name, Split = k, MSE = mse_val, ACC = acc, F1 = f1
    ))
    
    # 3f. Fast Global SHAP
    predict_fun <- function(object, newdata) {
      # CRITICAL FIX: $unsqueeze(2) added here for the background evaluations
      batch_tensor <- torch_tensor(as.matrix(newdata), dtype = torch_float(), device = device)$unsqueeze(2)
      with_no_grad({ pred <- as.numeric(object(batch_tensor)$cpu()) })
      return(pred)
    }
    x_tr_mat <- as.matrix(dataTrain[, vari])
    x_te_mat <- as.matrix(dataTest[, vari])
    colnames(x_te_mat) <- vari
    
    bg_idx <- sample(nrow(x_tr_mat), 500)
    te_idx <- sample(nrow(x_te_mat), 2000)
    
    shap_vals <- fastshap::explain(
      object = model, X = x_tr_mat[bg_idx, ], newdata = x_te_mat[te_idx, ],
      nsim = 50, pred_wrapper = predict_fun, shap_only = FALSE
    )
    
    global_imp <- colMeans(abs(as.matrix(shap_vals$shapley_values)))
    for (feat in names(global_imp)) {
      shap_collection <- rbind(shap_collection, data.frame(
        Scenario = sim_name, Split = k, Feature = feat, SHAP_Mean_Abs = as.numeric(global_imp[feat])
      ))
    }
  }
}

# --- 4. STATISTICAL AGGREGATION ---
cat("\n\n==================================================\n")
cat("FINAL ROBUSTNESS RESULTS (MEAN ± 95% CI)\n")
cat("K-fold cross validation LSTM70SPAT\n")
cat("==================================================\n")

final_metrics <- metrics_collection %>%
  group_by(Scenario) %>%
  summarize(
    MSE_Mean = mean(MSE), MSE_CI = 1.96 * (sd(MSE) / sqrt(n())),
    ACC_Mean = mean(ACC), ACC_CI = 1.96 * (sd(ACC) / sqrt(n())),
    F1_Mean  = mean(F1),  F1_CI  = 1.96 * (sd(F1)  / sqrt(n()))
  ) %>%
  mutate(
    MSE_Format = sprintf("%.4f ± %.4f", MSE_Mean, MSE_CI),
    ACC_Format = sprintf("%.3f ± %.3f", ACC_Mean, ACC_CI),
    F1_Format  = sprintf("%.3f ± %.3f", F1_Mean, F1_CI)
  ) %>%
  select(Scenario, MSE_Format, ACC_Format, F1_Format)

print(final_metrics)

cat("\n--- STABLE FEATURE IMPORTANCE ---\n")
final_shap <- shap_collection %>%
  group_by(Scenario, Feature) %>%
  summarize(
    SHAP_Mean = mean(SHAP_Mean_Abs), 
    SHAP_CI = 1.96 * (sd(SHAP_Mean_Abs) / sqrt(n())),
    .groups = 'drop'
  ) %>%
  arrange(Scenario, desc(SHAP_Mean)) %>%
  mutate(Importance_Format = sprintf("%.4f ± %.4f", SHAP_Mean, SHAP_CI)) %>%
  select(Scenario, Feature, Importance_Format)

print(final_shap)


# --- NEW: EXPORT SHAP DATA FOR PLOTTING ---
summary_export_data <- list(
  final_shap = final_shap,
  metrics = final_metrics
)

metrics_summary_file <- paste0(model_name,"_Kfold_Training_Summary_Table",".rds")
saveRDS(summary_export_data, file = metrics_summary_file)

message("Metrics summary saved to: ", metrics_summary_file)

