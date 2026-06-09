#R

# Temporal forecasting TTI 2026.05.26
# LSTM 
# add Secchi to RS 05.20

library(torch)
library(terra)
library(dplyr)
library(fastshap)
library(shapviz)
library(ggplot2)
library(Metrics)

# --- 1. SETUP & CUSTOM CHANNELS ---

device <- ifelse(cuda_is_available(), "cuda", "cpu")
torch_device(device)
print(paste("Device confirmed:", device))

if (cuda_is_available()) {
  gc(); cuda_empty_cache() #
}

print(paste("Training model LSTM60TIME"))

model_name = "LSTM60TIME"
# specify run_date today when running
run_date = Sys.Date()

ntime <- 14
ntrain = 12
  
target_var <- "TTI"
tt_threshold <- 0.71  # Final regulatory threshold

batch_size <- 2048
num_epochs <- 250
patience <- 20
learning_rate <- 0.002

# Custom R6 class layer to bridge standard list matrix output to 3D tensor stream
select_lstm_output <- nn_module(
  "select_lstm_output",
  initialize = function() {},
  forward = function(x) { return(x[[1]]) }
)

# Feature sets (Optimized: DIN:DIP excluded)
var_com = list(
  MSFD_Baseline  = c("Nit", "Pho", "Chla", "DO2_mg", "Secchi"), 
  Remote_Sensing = c("Temp", "Chla", "Depth", "Secchi"), 
  Comprehensive  = c("Nit", "Pho", "Chla", "DO2_mg", "Secchi", "Temp", "Depth")
)

# Automated Summary metrics storage framework
forecast_metrics_summary <- data.frame(
  model = model_name,
  Scenario = character(),
  MSE_TTI = numeric(),
  GES_Accuracy = numeric(),
  F1_Score = numeric(),
  stringsAsFactors = FALSE
)

# --- 2. GRID EXTRACTION ---
rfile <- './GES_Baltic_5year_ALLE_Combined_4D.nc'
sds_input <- terra::sds(rfile, guessCRS=TRUE)

Depth_rast <- sds_input$Depth
ii.good <- which(!is.na(values(Depth_rast[[1]]))) 
ngoodp <- length(ii.good)  # 29,502 coordinates

extracted_list <- list()
for (var_name in names(sds_input)) {
  extracted_list[[var_name]] <- as.vector(values(sds_input[[var_name]])[ii.good, ])
}
dataall <- as.data.frame(extracted_list)

# --- 3. CHRONOLOGICAL MATRIX BOUNDARY SPLICING ---
idx_2010 <- ngoodp * 12
idx_2015 <- ngoodp * 13
idx_2020 <- ngoodp * 14

dataTrain <- dataall[1:idx_2010, ]
dataVal   <- dataall[(idx_2010 + 1):idx_2015, ]
dataTest  <- dataall[(idx_2015 + 1):idx_2020, ]

# --- 4. BASAL TRAINING SCALE APPLICATION ---
# Compute metrics exclusively from baseline epochs to dodge leakage anomalies
predictors <- unique(unlist(var_com))
for (p in predictors) {
  mean_train <- mean(dataTrain[[p]], na.rm = TRUE)
  sd_train   <- sd(dataTrain[[p]], na.rm = TRUE)
  
  dataTrain[[p]] <- (dataTrain[[p]] - mean_train) / sd_train
  dataVal[[p]]   <- (dataVal[[p]] - mean_train) / sd_train
  dataTest[[p]]  <- (dataTest[[p]] - mean_train) / sd_train
}

# --- 5. LOOP OVER SCENARIOS ---
for (simul in seq_along(var_com)) {
  
  if (cuda_is_available()) {
    gc(); cuda_empty_cache() #
  }
  
  sim_name <- names(var_com)[simul]
  vari <- var_com[[simul]]
  nvari <- length(vari)
  
  print(paste("=================================================="))
  print(paste("Executing Chronological LSTM Forecast for:", sim_name))
  
  model_file_name.rt = paste0(model_name,'_',sim_name,'_',run_date,'.rt')
  
  # Inject 3D structural dimension sequence: [Batch, Sequence_Length (1), Features]
  x_train <- torch_tensor(as.matrix(dataTrain[, vari]), dtype=torch_float(), device=device)$unsqueeze(2)
  y_train <- torch_tensor(as.matrix(dataTrain[[target_var]]), dtype=torch_float(), device=device)
  
  x_val   <- torch_tensor(as.matrix(dataVal[, vari]), dtype=torch_float(), device=device)$unsqueeze(2)
  y_val   <- torch_tensor(as.matrix(dataVal[[target_var]]), dtype=torch_float(), device=device)
  
  x_test  <- torch_tensor(as.matrix(dataTest[, vari]), dtype=torch_float(), device=device)$unsqueeze(2)
  y_test  <- torch_tensor(as.matrix(dataTest[[target_var]]), dtype=torch_float(), device=device)
  
  # Standard Sequential Architecture Setup
  model <- nn_sequential(
    nn_lstm(input_size = nvari, hidden_size = 128, batch_first = TRUE),
    select_lstm_output(),
    nn_dropout(p = 0.2),
    
    nn_lstm(input_size = 128, hidden_size = 64, batch_first = TRUE),
    select_lstm_output(),
    nn_dropout(p = 0.2),
    
    # Crucial adjustment: flatten dimension boundary layer to yield standard [Batch, 1] output
    nn_flatten(start_dim = 2, end_dim = 3), 
    nn_linear(64, 1)
  )
  model$to(device=device)
  
  criterion <- nn_mse_loss()
  optimizer <- optim_adam(model$parameters, lr = learning_rate)
  
  n_train_rows <- nrow(dataTrain)
  n_val_rows   <- nrow(dataVal)
  
  train_batches <- 1:((n_train_rows - 1) %/% batch_size + 1)
  val_batches   <- 1:((n_val_rows - 1) %/% batch_size + 1)
  
  best_loss <- Inf
  best_model_state <- NULL
  epochs_no_improvement <- 0
  
  for (epoch in 1:num_epochs) {
    model$train()
    shuffled_indices <- sample(1:n_train_rows)
    epoch_loss <- 0
    
    for (j in train_batches) {
      start_index <- (j - 1) * batch_size + 1
      end_index   <- min(j * batch_size, n_train_rows)
      current_indices <- shuffled_indices[start_index:end_index]
      
      batch_x <- x_train[current_indices, , ]
      batch_y <- y_train[current_indices]$view(c(-1, 1))
      
      optimizer$zero_grad()
      outputs <- model(batch_x)
      loss <- criterion(outputs, batch_y)
      loss$backward()
      optimizer$step()
      epoch_loss <- epoch_loss + loss$item()
    }
    
    model$eval()
    val_loss <- 0
    with_no_grad({
      for (j in val_batches) {
        start_index <- (j - 1) * batch_size + 1
        end_index   <- min(j * batch_size, n_val_rows)
        
        batch_x <- x_val[start_index:end_index, , ]
        batch_y <- y_val[start_index:end_index]$view(c(-1, 1))
        
        val_outputs <- model(batch_x)
        loss <- criterion(val_outputs, batch_y)
        val_loss <- val_loss + loss$item()
      }
    })
    
    avg_train_loss <- epoch_loss / length(train_batches)
    avg_val_loss   <- val_loss / length(val_batches)
    
    cat(sprintf("Epoch: %d | Train MSE: %.4f | 2015 Val MSE: %.4f\r", epoch, avg_train_loss, avg_val_loss))
    
    if (avg_val_loss < best_loss) {
      best_loss <- avg_val_loss
      epochs_no_improvement <- 0
      best_model_state <- lapply(model$state_dict(), function(x) x$cpu()$clone())
    } else {
      epochs_no_improvement <- epochs_no_improvement + 1
    }
    
    if (epochs_no_improvement >= patience) {
      cat(sprintf("\nConvergence baseline plateaued. Early stopping triggered at epoch %d.\n", epoch))
      break
    }
  }
  
  if (!is.null(best_model_state)) {
    model$load_state_dict(best_model_state)
    model$to(device=device)
  }
  
  ## save the nodel, it makes also sense to predict for all data
  print(paste("Save model:",model_file_name.rt))
  torch_save(model, path = model_file_name.rt)
  
  # --- 6. 2020 OUT-OF-SAMPLE FORECAST EVALUATION ---
  model$eval()

  with_no_grad({ predictions_raw <- model(x_test) })
  prob_preds <- as.numeric(predictions_raw$cpu())
  true_tti   <- as.numeric(y_test$cpu())
  
  pred_ges <- ifelse(prob_preds > tt_threshold, 1, 0)
  true_ges <- ifelse(true_tti > tt_threshold, 1, 0)
  
  # Accurate calculation metrics 
  TP <- sum(pred_ges == 1 & true_ges == 1)
  FP <- sum(pred_ges == 1 & true_ges == 0)
  FN <- sum(pred_ges == 0 & true_ges == 1)
  TN <- sum(pred_ges == 0 & true_ges == 0)
  
  accuracy_val <- (TP + TN) / (TP + TN + FP + FN)
  precision    <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
  recall       <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
  f1_score     <- ifelse((precision + recall) == 0, 0, 2 * (precision * recall) / (precision + recall))
  mse_val      <- Metrics::mse(true_tti, prob_preds)
  
  # Save clean metrics array reference
  forecast_metrics_summary <- rbind(forecast_metrics_summary, data.frame(
    Scenario = sim_name,
    MSE_TTI = round(mse_val, 5),
    GES_Accuracy = round(accuracy_val, 3),
    F1_Score = round(f1_score, 3)
  ))
  
  # --- 7. SHAP EXPLANATIONS PIPELINE ---
  message(paste("Calculating Forecast SHAP values for", sim_name, "..."))
  x_train_matrix <- as.matrix(dataTrain[, vari])
  x_test_matrix  <- as.matrix(dataTest[, vari])
  colnames(x_test_matrix) <- vari
  
  predict_fun <- function(object, newdata) {
    batch_tensor <- torch_tensor(as.matrix(newdata), dtype = torch_float(), device = device)$unsqueeze(2)
    with_no_grad({ pred <- as.numeric(object(batch_tensor)$cpu()) })
    return(pred)
  }
  
  set.seed(123)
  bg_idx   <- sample(nrow(x_train_matrix), min(500, nrow(x_train_matrix)))
  test_idx <- sample(nrow(x_test_matrix), min(2000, nrow(x_test_matrix)))
  
  shap_values <- fastshap::explain(
    object = model,
    X = x_train_matrix[bg_idx, ], 
    newdata = x_test_matrix[test_idx, ],
    nsim = 50,
    pred_wrapper = predict_fun,
    shap_only = FALSE
  )
  
  # --- NEW: EXPORT SHAP DATA FOR PLOTTING ---
  shap_export_data <- list(
    shap_vals = shap_values,
    feature_data = x_test_matrix[test_idx, ]
  )
  # 
  export_filename <- paste0("SHAP_data_",model_name,"_",sim_name,".rds")
  saveRDS(shap_export_data, file = export_filename)
  
  message("SHAP data saved to: ", export_filename)
  
  # Beeswarm Summary Plot mapping
  shp <- shapviz(shap_values, X = x_test_matrix[test_idx, ])
  p_bee <- sv_importance(shp, kind = "beeswarm", max_display = nvari, show_numbers = TRUE) + 
    theme_minimal() + labs(title = paste("LSTM Forecast SHAP Distribution (2020):", sim_name))
  
  # High-quality Rocket Bar chart matrix layout
  raw_shap_matrix <- as.matrix(shap_values$shapley_values)
  global_importance <- colMeans(abs(raw_shap_matrix))
  
  importance_df <- data.frame(
    feature = names(global_importance),
    mean_abs_shap = as.numeric(global_importance)
  ) %>% arrange(mean_abs_shap) %>% mutate(feature = factor(feature, levels = feature))
  
  p_bar <- ggplot(importance_df, aes(x = mean_abs_shap, y = feature, fill = mean_abs_shap)) +
    geom_col() + geom_text(aes(label = round(mean_abs_shap, 4)), hjust = -0.1, size = 4) +
    scale_fill_viridis_c(option = "rocket", direction = -1, guide = "none") +
    theme_minimal() + theme(axis.text.y = element_text(size = 12, face = "bold"), panel.grid.major.y = element_blank()) +
    labs(title = paste("LSTM60 Forecast Global Feature Importance:", sim_name), x = "Mean Absolute SHAP Value (Impact on TTI)", y = NULL) +
    expand_limits(x = max(importance_df$mean_abs_shap) * 1.15)
  
  print(p_bee)
  print(p_bar)
  
  #ggsave(filename = paste0(model_name,"_Forecast_SHAP_Beeswarm_", sim_name, ".png"), plot = p_bee, width = 8, height = 6, bg = "white")
  #ggsave(filename = paste0(model_name,"_Forecast_SHAP_Bar_", sim_name, ".png"), plot = p_bar, width = 8, height = 6, bg = "white")
}

# --- 8. COMPILER MATRIX TERMINAL MONITOR ---
print("==================================================")
print("FINAL TEMPORAL FORECAST METRICS SUMMARY (LSTM60TIME):")
print(forecast_metrics_summary)

metrics_summary$Model <- model_name

metrics_summary_file <- paste0(model_name,"_Training_Summary_Table",".rds")
saveRDS(metrics_summary, file = metrics_summary_file)
message("Metrics summary saved to: ", metrics_summary_file)

library(gt)

metrics_summary %>% gt





