#R

# 2026.05.26 Temporal forecasting loop

library(torch)
library(terra)
library(dplyr)
library(fastshap)
library(shapviz)
library(ggplot2)
library(Metrics)

# --- 1. CONFIGURATION & TARGET SETUP ---
# --- 1. SETUP & TARGET SETS ---

device <- ifelse(cuda_is_available(), "cuda", "cpu")
torch_device(device)
print(paste("Device forced to:", device))

if (cuda_is_available()) {
  gc(); cuda_empty_cache() #
}

ntime <- 14
ntrain <- 12

target_var <- "TTI"
tt_threshold <- 0.71  # Updated published threshold

batch_size <- 1024
batch_size = 2048 # batch size for training, bigger is faster
num_epochs <- 250
patience <- 20
learning_rate <- 0.002
# specify run_date today when running
run_date = Sys.Date()

model_name="MLP60TIME"

# Targeted feature sets (DIN:DIP ratio removed cleanly)
var_com = list(
  MSFD_Baseline = c("Nit", "Pho", "Chla", "DO2_mg", "Secchi"), 
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

# --- 2. DATA EXTRACTION ---
rfile <- './GES_Baltic_5year_ALLE_Combined_4D.nc'
sds_input <- terra::sds(rfile, guessCRS=TRUE)

Depth_rast <- sds_input$Depth
ii.good <- which(!is.na(values(Depth_rast[[1]]))) 
ngoodp <- length(ii.good)  # 29,502 water grid points

extracted_list <- list()
for (var_name in names(sds_input)) {
  v <- values(sds_input[[var_name]])
  extracted_list[[var_name]] <- as.vector(v[ii.good, ])
}
dataall <- as.data.frame(extracted_list)

# --- 3. CHRONOLOGICAL DATA SPLITTING ---
# Time steps: 1-12 (1955-2010), 13 (2015), 14 (2020)
idx_2010 <- ngoodp * 12
idx_2015 <- ngoodp * 13
idx_2020 <- ngoodp * 14

dataTrain <- dataall[1:idx_2010, ]
dataVal   <- dataall[(idx_2010 + 1):idx_2015, ]
dataTest  <- dataall[(idx_2015 + 1):idx_2020, ]

# --- 4. DATA SCALING (Rigorous Historical Baseline Calculation) ---
predictors <- unique(unlist(var_com))
for (p in predictors) {
  mean_train <- mean(dataTrain[[p]], na.rm = TRUE)
  sd_train   <- sd(dataTrain[[p]], na.rm = TRUE)
  
  dataTrain[[p]] <- (dataTrain[[p]] - mean_train) / sd_train
  dataVal[[p]]   <- (dataVal[[p]] - mean_train) / sd_train
  dataTest[[p]]  <- (dataTest[[p]] - mean_train) / sd_train
}

cat(sprintf("Slicing Complete.\nTraining (1955-2010): %d\nValidation (2015): %d\nForecasting Target (2020): %d\n", 
            nrow(dataTrain), nrow(dataVal), nrow(dataTest)))

# --- 5. TEMPORAL FORECAST MODEL PIPELINE ---
for (simul in seq_along(var_com)) {
  
  sim_name <- names(var_com)[simul]
  vari <- var_com[[simul]]
  nvari <- length(vari)
  
  print(paste("--------------------------------------------------"))
  print(paste("Running Scenario:", sim_name))
  
  model_file_name.rt = paste0(model_name,'_',sim_name,'_',run_date,'.rt')
  
  # Set up tensors
  x_train <- torch_tensor(as.matrix(dataTrain[, vari]), dtype=torch_float(), device=device)
  y_train <- torch_tensor(as.matrix(dataTrain[[target_var]]), dtype=torch_float(), device=device)
  
  x_val   <- torch_tensor(as.matrix(dataVal[, vari]), dtype=torch_float(), device=device)
  y_val   <- torch_tensor(as.matrix(dataVal[[target_var]]), dtype=torch_float(), device=device)
  
  x_test  <- torch_tensor(as.matrix(dataTest[, vari]), dtype=torch_float(), device=device)
  y_test  <- torch_tensor(as.matrix(dataTest[[target_var]]), dtype=torch_float(), device=device)
  
  # Model Setup (Sequential Regression Network)
  model <- nn_sequential(
    nn_linear(nvari, 128),
    nn_relu(),
    nn_dropout(p = 0.2),
    nn_linear(128, 64),
    nn_relu(),
    nn_dropout(p = 0.2),
    nn_linear(64, 1)
  )
  model$to(device=device)
  
  criterion <- nn_mse_loss()
  optimizer <- optim_adam(model$parameters, lr = learning_rate)
  
  n_train <- nrow(dataTrain)
  n_val   <- nrow(dataVal)
  train_batches <- 1:((n_train - 1) %/% batch_size + 1)
  val_batches   <- 1:((n_val - 1) %/% batch_size + 1)
  
  best_loss <- Inf
  best_model_state <- NULL
  epochs_no_improvement <- 0
  
  # Training loop with validation early stopping
  for (epoch in 1:num_epochs) {
    model$train()
    shuffled_indices <- sample(1:n_train)
    epoch_loss <- 0
    
    for (j in train_batches) {
      start_index <- (j - 1) * batch_size + 1
      end_index   <- min(j * batch_size, n_train)
      current_indices <- shuffled_indices[start_index:end_index]
      
      batch_x <- x_train[current_indices, ]
      batch_y <- y_train[current_indices]$view(c(-1, 1))
      
      optimizer$zero_grad()
      outputs <- model(batch_x)
      loss <- criterion(outputs, batch_y)
      loss$backward()
      optimizer$step()
      epoch_loss <- epoch_loss + loss$item()
    }
    
    # Evaluate against the 2015 Validation block for early stopping
    model$eval()
    val_loss <- 0
    with_no_grad({
      for (j in val_batches) {
        start_index <- (j - 1) * batch_size + 1
        end_index   <- min(j * batch_size, n_val)
        
        batch_x <- x_val[start_index:end_index, ]
        batch_y <- y_val[start_index:end_index]$view(c(-1, 1))
        
        val_outputs <- model(batch_x)
        loss <- criterion(val_outputs, batch_y)
        val_loss <- val_loss + loss$item()
      }
    })
    
    avg_train_loss <- epoch_loss / length(train_batches)
    avg_val_loss   <- val_loss / length(val_batches)
    
    cat(sprintf("Epoch: %d | Historical Train MSE: %.4f | 2015 Val MSE: %.4f\r", 
                epoch, avg_train_loss, avg_val_loss))
    
    if (avg_val_loss < best_loss) {
      best_loss <- avg_val_loss
      epochs_no_improvement <- 0
      best_model_state <- lapply(model$state_dict(), function(x) x$cpu()$clone())
    } else {
      epochs_no_improvement <- epochs_no_improvement + 1
    }
    
    if (epochs_no_improvement >= patience) {
      cat(sprintf("\nConvergence ceiling hit. Early stopping triggered at epoch %d.\n", epoch))
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
  
  # --- 6. OUT-OF-SAMPLE 2020 FORECAST METRICS ---
  model$eval()
  with_no_grad({
    predictions_raw <- model(x_test)
  })
  prob_preds <- as.numeric(predictions_raw$cpu())
  true_tti   <- as.numeric(y_test$cpu())
  
  pred_ges <- ifelse(prob_preds > tt_threshold, 1, 0)
  true_ges <- ifelse(true_tti > tt_threshold, 1, 0)
  
  # Manual confusion evaluation logic
  
  # Manual Confusion Matrix & F1 Score
  TP <- sum(pred_ges == 1 & true_ges == 1)
  FP <- sum(pred_ges == 1 & true_ges == 0)
  FN <- sum(pred_ges == 0 & true_ges == 1)
  TN <- sum(pred_ges == 0 & true_ges == 0)
  
  accuracy_val <- (TP + TN) / (TP + TN + FP + FN)
  precision    <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
  recall       <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
  f1_score     <- ifelse((precision + recall) == 0, 0, 2 * (precision * recall) / (precision + recall))
  mse_val      <- Metrics::mse(true_tti, prob_preds)
  
  cat(sprintf("\n>>> 2020 FORECAST RESULTS (%s):\nContinuous TTI MSE: %.5f\nTargeted Threshold Accuracy: %.3f\nF1-Score: %.3f\n\n", 
              sim_name, mse_val, accuracy_val, f1_score))
  
  # Store inside summary metrics data frame
  metrics_summary <- rbind(metrics_summary, data.frame(
    Scenario = sim_name,
    MSE_TTI = round(mse_val, 5),
    GES_Accuracy = round(accuracy_val, 3),
    F1_Score = round(f1_score, 3)
  ))
  
  # --- 7. SPATIO-TEMPORAL SHAP EXPLANATION ---
  message(paste("Calculating Forecast SHAP values for", sim_name, "..."))
  x_train_matrix <- as.matrix(dataTrain[, vari])
  x_test_matrix  <- as.matrix(dataTest[, vari])
  colnames(x_test_matrix) <- vari
  
  predict_fun <- function(object, newdata) {
    batch_tensor <- torch_tensor(as.matrix(newdata), dtype = torch_float(), device = device)
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
  export_filename <- paste0("SHAP_data_",model_name,"_",sim_name,".rds")
  saveRDS(shap_export_data, file = export_filename)
  message("SHAP data saved to: ", export_filename)
  
  # Visual 1: Shapviz Beeswarm Summary
  shp <- shapviz(shap_values, X = x_test_matrix[test_idx, ])
  p_bee <- sv_importance(shp, kind = "beeswarm", max_display = nvari, show_numbers = TRUE) + 
    theme_minimal() +
    theme(plot.title = element_text(face = "bold", hjust=0.5, size=16)) + 
    labs(title = paste("Forecast SHAP Distribution (2020):", sim_name))
  
  # Visual 2: Custom Rocket Gradient Global Summary
  raw_shap_matrix <- as.matrix(shap_values$shapley_values)
  global_importance <- colMeans(abs(raw_shap_matrix))
  
  importance_df <- data.frame(
    feature = names(global_importance),
    mean_abs_shap = as.numeric(global_importance)
  ) %>%
    arrange(mean_abs_shap) %>% 
    mutate(feature = factor(feature, levels = feature))
  
  p_bar <- ggplot(importance_df, aes(x = mean_abs_shap, y = feature, fill = mean_abs_shap)) +
    geom_col() +
    geom_text(aes(label = round(mean_abs_shap, 4)), hjust = -0.1, size = 4) +
    scale_fill_viridis_c(option = "rocket", direction = -1, guide = "none") +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5, size = 16),
      axis.text.y = element_text(size = 12, face = "bold"),
      panel.grid.major.y = element_blank()
    ) +
    labs(title = paste("Forecast Global Feature Importance:", sim_name),
         x = "Mean Absolute SHAP Value (Impact on TTI)", y = NULL) +
    expand_limits(x = max(importance_df$mean_abs_shap) * 1.15)
  
  print(p_bee)
  print(p_bar)
  
  #ggsave(filename = paste0(model_name,"_SHAP_Beeswarm_", sim_name, ".png"), plot = p_bee, width = 8, height = 6, bg = "white")
  #ggsave(filename = paste0(model_name,"_SHAP_Bar_", sim_name, ".png"), plot = p_bar, width = 8, height = 6, bg = "white")
}


# --- 7. FINAL COMPARATIVE OUTPUT ---
print("==================================================")
print("COMPILATION SUMMARY TABLE (MLP60 Temporal FORECAST):")
print(metrics_summary)

metrics_summary$Model <- model_name

metrics_summary_file <- paste0(model_name,"_Training_Summary_Table",".rds")
saveRDS(metrics_summary, file = metrics_summary_file)
message("Metrics summary saved to: ", metrics_summary_file)


library(gt)

metrics_summary %>% gt

