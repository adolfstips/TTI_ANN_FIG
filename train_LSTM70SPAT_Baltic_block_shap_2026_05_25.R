#

#R LSTM spatial extrapolation

# LSTM70SPAT with SHAP analysis

#2026.05.25 add Secchi to RS

library(torch)
library(terra)
library(dplyr)
library(fastshap)
library(shapviz)
library(ggplot2)
library(Metrics)

device <- ifelse(cuda_is_available(), "cuda", "cpu")
torch_device(device)
print(paste("Using device:", device))
if (cuda_is_available()) {
  gc(); cuda_empty_cache() #
}

ntime <- 14
t_timesteps <- 14

target_var <- "TTI"
tt_threshold <- 0.71

batch_size <- 1024
num_epochs <- 250
patience <- 20
learning_rate <- 0.002

dropout_rate <- 0.2

print(paste("Training model LSTM70SPAT:"))
# specify run_date today when running
run_date = Sys.Date()

model_name = "LSTM70SPAT"

# Register the selection layer
select_lstm_output <- nn_module(
  "select_lstm_output",
  initialize = function() {},
  forward = function(x) { return(x[[1]]) }
)

# Target configurations (DIN:DIP ratio removed)
var_com = list(
  MSFD_Baseline  = c("Nit", "Pho", "Chla", "DO2_mg", "Secchi")  , 
  Remote_Sensing = c("Temp", "Chla", "Depth", "Secchi"), 
  Comprehensive  = c("Nit", "Pho", "Chla", "DO2_mg", "Secchi", "Temp", "Depth")
)

# Continuous Metrics collection table
metrics_summary <- data.frame(
  Scenario = character(),
  MSE_TTI = numeric(),
  GES_Accuracy = numeric(),
  F1_Score = numeric(),
  stringsAsFactors = FALSE
)

# --- 2. DATA READING & SPATIAL BLOCK GENERATION ---
rfile <- './GES_Baltic_5year_ALLE_Combined_4D.nc'
sds_input <- terra::sds(rfile, guessCRS=TRUE)

Depth_rast <- sds_input$Depth
ii.good <- which(!is.na(values(Depth_rast[[1]]))) 
ngoodp <- length(ii.good)

rc <- terra::rowColFromCell(Depth_rast[[1]], ii.good)
block_size <- 20
spatial_blocks <- paste(ceiling(rc[,1] / block_size), ceiling(rc[,2] / block_size), sep="_")

extracted_list <- list()
for (var_name in names(sds_input)) {
  extracted_list[[var_name]] <- as.vector(values(sds_input[[var_name]])[ii.good, ])
}
dataall <- as.data.frame(extracted_list)
dataall$Block_ID <- rep(spatial_blocks, ntime)

# --- 3. SCALING & SPLITTING ---
predictors <- unique(unlist(var_com))
for (p in predictors) {
  dataall[[p]] <- scale(dataall[[p]])
}

ii_train_end <- ngoodp * t_timesteps
data_subset <- dataall[1:ii_train_end, ]

set.seed(123)
unique_blocks <- unique(data_subset$Block_ID)
train_blocks <- sample(unique_blocks, size = length(unique_blocks) * 0.60)

dataTrain <- data_subset %>% filter(Block_ID %in% train_blocks)
dataTest  <- data_subset %>% filter(!(Block_ID %in% train_blocks))

n_train <- nrow(dataTrain)
n_val   <- nrow(dataTest)

# --- 4. SEQUENTIAL LSTM PIPELINE ---
for (simul in seq_along(var_com)) {
  
  if (cuda_is_available()) {
    gc(); cuda_empty_cache() #
  }
  
  sim_name <- names(var_com)[simul]
  vari <- var_com[[simul]]
  nvari <- length(vari)
  
  print(paste("--------------------------------------------------"))
  print(paste("Running Scenario:", sim_name))
  
  model_file_name.rt = paste0(model_name,'_',sim_name,'_',run_date,'.rt')
  
    # Reshape data tensors into 3D structure: [Batch, Sequence_Length (1), Features]
  x_train <- torch_tensor(as.matrix(dataTrain[, vari]), dtype=torch_float(), device=device)$unsqueeze(2)
  y_train <- torch_tensor(as.matrix(dataTrain[[target_var]]), dtype=torch_float(), device=device)
  
  x_test  <- torch_tensor(as.matrix(dataTest[, vari]), dtype=torch_float(), device=device)$unsqueeze(2)
  y_test  <- torch_tensor(as.matrix(dataTest[[target_var]]), dtype=torch_float(), device=device)
 
  # Define the custom layer to extract the main tensor from LSTM output list
  # Pure nn_sequential regression LSTM model architecture
  model <- nn_sequential(
    nn_lstm(input_size = nvari, hidden_size = 128, batch_first = TRUE),
    select_lstm_output(),
    nn_dropout(p = 0.2),
    
    nn_lstm(input_size = 128, hidden_size = 64, batch_first = TRUE),
    select_lstm_output(),
    nn_dropout(p = 0.2),
    # Flatten out sequence dimension to pass to Linear Regression Layer
    # [Batch, 1, 64] -> [Batch, 64]
    nn_flatten(start_dim = 2, end_dim = 3), 
    nn_linear(64, 1)
  )
  
  model$to(device=device)
  
  criterion <- nn_mse_loss()
  optimizer <- optim_adam(model$parameters, lr = learning_rate)
  
  train_batches <- 1:((n_train - 1) %/% batch_size + 1)
  val_batches   <- 1:((n_val - 1) %/% batch_size + 1)
  
  best_loss <- Inf
  best_model_state <- NULL
  epochs_no_improvement <- 0
  
  for (epoch in 1:num_epochs) {
    model$train()
    shuffled_indices <- sample(1:n_train)
    epoch_loss <- 0
    
    for (j in train_batches) {
      start_index <- (j - 1) * batch_size + 1
      end_index   <- min(j * batch_size, n_train)
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
        end_index   <- min(j * batch_size, n_val)
        
        batch_x <- x_test[start_index:end_index, , ]
        batch_y <- y_test[start_index:end_index]$view(c(-1, 1))
        
        val_outputs <- model(batch_x)
        loss <- criterion(val_outputs, batch_y)
        val_loss <- val_loss + loss$item()
      }
    })
    
    avg_train_loss <- epoch_loss / length(train_batches)
    avg_val_loss   <- val_loss / length(val_batches)
    
    cat(sprintf("Epoch: %d | Train MSE: %.4f | Val MSE: %.4f\r", epoch, avg_train_loss, avg_val_loss))
    
    if (avg_val_loss < best_loss) {
      best_loss <- avg_val_loss
      epochs_no_improvement <- 0
      best_model_state <- lapply(model$state_dict(), function(x) x$cpu()$clone())
    } else {
      epochs_no_improvement <- epochs_no_improvement + 1
    }
    
    if (epochs_no_improvement >= patience) {
      cat(sprintf("\nEarly stopping triggered at epoch %d.\n", epoch))
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
  
  # --- 5. EVALUATION METRICS ---
  model$eval()
  with_no_grad({ predictions_raw <- model(x_test) })
  prob_preds <- as.numeric(predictions_raw$cpu())
  true_tti   <- as.numeric(y_test$cpu())
  
  pred_ges <- ifelse(prob_preds > tt_threshold, 1, 0)
  true_ges <- ifelse(true_tti > tt_threshold, 1, 0)
  
  TP <- sum(pred_ges == 1 & true_ges == 1)
  FP <- sum(pred_ges == 1 & true_ges == 0)
  FN <- sum(pred_ges == 0 & true_ges == 1)
  TN <- sum(pred_ges == 0 & true_ges == 0)
  
  accuracy_val <- (TP + TN) / (TP + TN + FP + FN)
  precision    <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
  recall       <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
  f1_score     <- ifelse((precision + recall) == 0, 0, 2 * (precision * recall) / (precision + recall))
  mse_val      <- Metrics::mse(true_tti, prob_preds)
  
  # Store inside summary metrics data frame
  metrics_summary <- rbind(metrics_summary, data.frame(
    Scenario = sim_name,
    MSE_TTI = round(mse_val, 5),
    GES_Accuracy = round(accuracy_val, 3),
    F1_Score = round(f1_score, 3)
  ))
  
  # --- 6. PURE NN_SEQUENTIAL SHAP OVERRIDE ---
  message(paste("Calculating Forecast SHAP values for", sim_name, "..."))
  x_train_matrix <- as.matrix(dataTrain[, vari])
  x_test_matrix  <- as.matrix(dataTest[, vari])
  colnames(x_test_matrix) <- vari
  
  # Fastshap prediction wrapper handles the 3D tensor mapping natively
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
  
  print(paste("Save shap values :",model_file_name.Rdata))
  # --- NEW: EXPORT SHAP DATA FOR PLOTTING ---
  shap_export_data <- list(
    shap_vals = shap_values,
    feature_data = x_test_matrix[test_idx, ]
  )
  
  # 
  # 
  export_filename <- paste0("SHAP_data_LSTM70SPAT_", sim_name, ".rds")
  saveRDS(shap_export_data, file = export_filename)
  message("SHAP data saved to: ", export_filename)
  
   # Shapviz plots
  shp <- shapviz(shap_values, X = x_test_matrix[test_idx, ])
  p_bee <- sv_importance(shp, kind = "beeswarm", max_display = nvari, show_numbers = TRUE) + 
    theme_minimal() + labs(title = paste("LSTM70 Spatial SHAP Distribution:", sim_name))
  
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
    labs(title = paste("LSTM70 Spatial Global Feature Importance:", sim_name), x = "Mean Absolute SHAP Value (Impact on TTI)", y = NULL) +
    expand_limits(x = max(importance_df$mean_abs_shap) * 1.15)
  
  print(p_bee)
  print(p_bar)
  
 # ggsave(filename = paste0(model_name,"_SHAP_Beeswarm_", sim_name, ".png"), plot = p_bee, width = 8, height = 6, bg = "white")
 # ggsave(filename = paste0(model_name,"_SHAP_Bar_", sim_name, ".png"), plot = p_bar, width = 8, height = 6, bg = "white")

  }

# --- 7. FINAL COMPARATIVE OUTPUT ---
print("==================================================")
print("COMPILATION SUMMARY TABLE (LSTM70 SPATIAL EXTRAPOLATION):")
print(metrics_summary)

metrics_summary_file <- paste0(model_name,"_Training_Summary_Table",".rds")

metrics_summary$Model <- model_name

saveRDS(metrics_summary, file = metrics_summary_file)
message("Metrics summary saved to: ", metrics_summary_file)


library(gt)

metrics_summary %>% gt

