#R

#2026.05.26

## use spatial block subsampling
# MLP70SPAT
# train the 3 modles

library(torch)
library(terra)
library(dplyr)
library(caret)
library(Metrics)
library(fastshap)
library(viridis)
library(shapviz)
library(ggplot2)

device <- ifelse(cuda_is_available(), "cuda", "cpu")
torch_device(device)
print(paste("Device:", device))

if (cuda_is_available()) {
  gc(); cuda_empty_cache() #
}

set.seed(123)

plot_block_map = FALSE

# specify run_date today when running
run_date = Sys.Date()

## set train to test ratio  use 0.6 ??
trainr = 0.6

## according to luz could be higher, but not working!
learning_rate = 0.002

### select max number of epochs - 
num_epochs = 250

patience = 20 # use 10% max epochs, number of cycles to wait before early stopping

# Train the model
batch_size = 2048 # batch size for training, bigger is faster?

ntime <- 14
t_timesteps <- 14

model_name = "MLP70SPAT"

# Define exact variable names based on your NetCDF
target_var <- "TTI" # Update if your NetCDF uses 'TTind'
tt_threshold <- 0.71 # New published GES threshold

# Streamlined targeted variable combinations
var_com = list(
  MSFD_Baseline = c("Nit", "Pho", "Chla", "DO2_mg", "Secchi"), 
  Remote_Sensing = c("Temp", "Chla", "Depth","Secchi"), 
  Comprehensive = c("Nit", "Pho", "Chla", "DO2_mg", "Secchi", "Temp", "Depth")
)

# Continuous Metrics collection table
metrics_summary <- data.frame(
  Scenario = character(),
  MSE_TTI = numeric(),
  GES_Accuracy = numeric(),
  F1_Score = numeric(),
  stringsAsFactors = FALSE
)

# --- 2. DATA READING & SPATIAL BLOCK PARTITIONING ---
rfile <- './GES_Baltic_5year_ALLE_Combined_4D.nc'
sds_input <- terra::sds(rfile, guessCRS=TRUE)

Depth_rast <- sds_input$Depth
ii.good <- which(!is.na(values(Depth_rast[[1]]))) # Valid water pixels
ngoodp <- length(ii.good)

# Generate Spatial Blocks (e.g., 20x20 pixels) to prevent spatial autocorrelation
rc <- terra::rowColFromCell(Depth_rast[[1]], ii.good)
block_size <- 20
block_row <- ceiling(rc[,1] / block_size)
block_col <- ceiling(rc[,2] / block_size)
spatial_blocks <- paste(block_row, block_col, sep="_")

# Extract data into a long-format dataframe
extracted_list <- list()
for (var_name in names(sds_input)) {
  v <- values(sds_input[[var_name]])
  extracted_list[[var_name]] <- as.vector(v[ii.good, ])
}

dataall <- as.data.frame(extracted_list)
all_varnams <- colnames(dataall)

# Replicate the spatial blocks for all time steps so each pixel retains its block ID
dataall$Block_ID <- rep(spatial_blocks, ntime)

# --- 3. SCALING & BLOCK-BASED SPLIT ---
# Scale predictors (excluding target variable and Block_ID)
predictors <- unlist(unique(var_com))
for (p in predictors) {
  if(p %in% all_varnams) {
    dataall[[p]] <- scale(dataall[[p]])
  } else {
    warning(paste("Variable", p, "not found in NetCDF. Check names!"))
  }
}

# Subset for training time steps
ii_train_end <- ngoodp * t_timesteps
data_subset <- dataall[1:ii_train_end, ]

# Perform Block-Based Train/Test Split (approx 60/40)
set.seed(123)
unique_blocks <- unique(data_subset$Block_ID)
train_blocks <- sample(unique_blocks, size = length(unique_blocks) * 0.60)

dataTrain <- data_subset %>% filter(Block_ID %in% train_blocks)
dataTest  <- data_subset %>% filter(!(Block_ID %in% train_blocks))

n_train <- nrow(dataTrain)
n_val   <- nrow(dataTest)

cat("Spatial Block Splitting Complete.\n")
cat("Training points:", n_train, "\nValidation points:", n_val, "\n")

if (plot_block_map) {
# --- VISUALIZE SPATIAL BLOCKS FOR SM ---
# 1. Create a blank template raster using the first time step
block_map <- sds_input$Depth[[1]]
terra::values(block_map) <- NA # Clear existing depths

# 2. Create a binary vector for the water points (1 = Train, 0 = Val)
# We only need the first 'ngoodp' rows since the spatial grid is static over time
is_train <- ifelse(dataall$Block_ID[1:ngoodp] %in% train_blocks, 1, 0)

# 3. Fill the water points in the raster
terra::values(block_map)[ii.good] <- is_train

# 4. Plot and save
png("Spatial_Block_Split_SM.png", width = 800, height = 1000, res = 150)

plot(block_map, col = c("lightblue", "darkorange"), 
     main = "Spatial Cross-Validation Blocks\n(Orange = Training, Blue = Validation)",
     legend = FALSE, axes = TRUE)

legend("bottomright", inset = c(0.02, 0.09), # Pushes it 5% right and 5% up
       legend = c("Training Data", "Validation Data"), 
       fill = c("darkorange", "lightblue"), bg = "white", cex = 0.9)

dev.off()

message("Spatial block map saved for SM.")
}


# --- 4. REGRESSION MODEL PREPARATION ---

# Example looping through your streamlined variable sets
for (simul in seq_along(var_com)) {
  
  start_time <- Sys.time()
  
  sim_name <- names(var_com)[simul]
  vari <- var_com[[simul]]
  nvari <- length(vari)
  
  print(paste("Running:", sim_name, "- Vars:", paste(vari, collapse=", ")))

  model_file_name = paste0(model_name,'_',sim_name,'_',run_date,'.rt')
  
  # Convert to Tensors for REGRESSION (Continuous Target)
  x_train <- torch_tensor(as.matrix(dataTrain[, vari]), dtype=torch_float(), device=device)
  y_train <- torch_tensor(as.matrix(dataTrain[[target_var]]), dtype=torch_float(), device=device)
  
  x_test <- torch_tensor(as.matrix(dataTest[, vari]), dtype=torch_float(), device=device)
  y_test <- torch_tensor(as.matrix(dataTest[[target_var]]), dtype=torch_float(), device=device)
  
  # Define the model (NO nn_sigmoid at the end for regression)
  model <- nn_sequential(
    nn_linear(nvari, 128),
    nn_relu(),
    nn_dropout(p = 0.2),
    nn_linear(128, 64),
    nn_relu(),
    nn_dropout(p = 0.2),
    nn_linear(64, 1) # Outputs a raw continuous TTI value
  )
  
  model$to(device=device)
  
  # Use Mean Squared Error for continuous target
  criterion <- nn_mse_loss()
  optimizer <- optim_adam(model$parameters, lr = 0.002)
  
  # ... [Proceed with standard training loop] ...
  
  train_batches = 1:((n_train - 1) %/% batch_size + 1)
  val_batches = 1:((n_val - 1) %/% batch_size + 1)
  
  num_val_batches = length(val_batches)
  num_train_batches = length(train_batches)
  
  best_model = NULL
  epochs_no_improvement <- 0
  best_loss <- Inf
  best_model_state <- NULL  # Store the best model's state dictionary
  lloss <- 0
  
  # Training loop
  
    for (epoch in 1: num_epochs) {
      ## number epochs
      
      model$train() # Set model to training mode
      epoch_loss <- 0
      
      # --- OPTIMIZATION: Shuffle training indices every epoch ---
      # This prevents the model from learning spatial patterns based on index order
      shuffled_indices <- sample(1:n_train)
      
      for (j in train_batches) {
        
        ###  train model
        start_index = (j - 1) * batch_size + 1
        end_index = min(j * batch_size, n_train ) # use min to prevent index out of bounds
        
        # Use the shuffled indices
        current_indices <- shuffled_indices[start_index:end_index]
        
        batch_x = x_train[start_index:end_index, ]
        #It transforms your target vector into a 2D column matrix.
        batch_y = y_train[start_index:end_index]$view(c(-1, 1))
        
        # Ensure batch_y is reshaped to [BatchSize, 1] to match model output
        #batch_x <- x_train[current_indices, ]
        #batch_y <- y_train[current_indices]$view(c(-1, 1))
        
        optimizer$zero_grad()
        
        outputs = model(batch_x)  ## the model!
        loss = criterion(outputs, batch_y)
        loss$backward()
        optimizer$step()
        epoch_loss <- epoch_loss + loss$item()
        
      } ## end of train batches
      
      avg_epoch_loss <- epoch_loss / num_train_batches
      
      # Validation, set model to validation mode!
      
      model$eval()
      
      val_loss <- 0
      
      coro::loop(for (j in val_batches) {
        ### val_batches
        
        ###  train model
        start_index = (j - 1) * batch_size + 1
        end_index = min(j * batch_size, n_val ) # use min to prevent index out of bounds
        
        batch_x = x_test[start_index:end_index, ]
        #batch_y = y_test[start_index:end_index]
        #It transforms your target vector into a 2D column matrix.
        batch_y <- y_test[start_index:end_index]$view(c(-1, 1))
        
        # Correct way to disable gradients in R:
        val_outputs <- with_no_grad(model(batch_x))  # <--- Key Change
        
        loss = criterion(val_outputs, batch_y)
        val_loss <- val_loss + loss$item()
        
      })
      
      avg_val_loss <- round(val_loss / num_val_batches,3)
      
      cat(paste(
        "Epoch:",epoch,
        "Train Loss:",round(avg_epoch_loss, 3),
        "Val Loss:",round(avg_val_loss, 3), "\r"
      ))
      
      # Early stopping
      if (avg_val_loss < best_loss) {
        best_loss <- avg_val_loss
        epochs_no_improvement <- 0
        # Save the best model (optional)
        # best_model_state <- model$state_dict()  # <--- Save the BEST model's state
        best_model_state <- lapply(model$state_dict(), function(x) x$cpu()$clone())
      } else {
        epochs_no_improvement <- epochs_no_improvement + 1
      }
      
      if (epochs_no_improvement >= patience) {
        cat("\n Early stopping triggered: ", epoch, "\n")
        break  # Exit the training loop
      }
      
    } # end of epochs
    
    ######### End of training
    end_time =  Sys.time()
    
    ##difftime(end_time,start_time,units="mins")
    sim_time = round(difftime(end_time,start_time,units="mins"), 3)
    print(paste("Time used: ",sim_time , 'mins'))
    
    lloss= round(best_loss,3)
    print(paste("Best Loss: ", lloss))
    
    # Load the best model's state after training is complete
    if (!is.null(best_model_state)) {
      model$load_state_dict(best_model_state)  # <--- Load the best state
      cat("Best model loaded.\n")
    } else {
      cat("No best model found, model from last epoch is used.\n")
    }
    
    ## save the nodel, it makes also sense to predict for all data
    print(paste("Save model:",model_file_name))
    torch_save(model, path = model_file_name)
    
    # --- 5. EVALUATION POST-TRAINING ---
    # To evaluate policy accuracy, you apply the threshold AFTER prediction
    
    model$eval()
    predictions_raw <- with_no_grad(model(x_test))
    prob_preds <- as.numeric(predictions_raw$cpu()) # Continuous TTI predictions
    true_tti <- as.numeric(y_test$cpu())
    
    # Re-calculate binary GES for accuracy/F1 based on the threshold
    pred_ges <- ifelse(prob_preds > tt_threshold, 1, 0)
    true_ges <- ifelse(true_tti > tt_threshold, 1, 0)
    
    # Manual Confusion Matrix & F1 Score
    TP <- sum(pred_ges == 1 & true_ges == 1)
    FP <- sum(pred_ges == 1 & true_ges == 0)
    FN <- sum(pred_ges == 0 & true_ges == 1)
    TN <- sum(pred_ges == 0 & true_ges == 0)
    
    accuracy_val <- (TP + TN) / (TP + TN + FP + FN)
    precision <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
    recall    <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
    f1_score  <- ifelse((precision + recall) == 0, 0, 2 * (precision * recall) / (precision + recall))
    
    mse_val <- Metrics::mse(true_tti, prob_preds)
    ## Better to rmse - same unit as variables!
    
    cat(sprintf("\n%s Results:\nMSE (TTI): %.4f | GES Acc: %.3f | F1: %.3f\n", 
                sim_name, mse_val, accuracy_val, f1_score))
    
    # Store inside summary metrics data frame
    metrics_summary <- rbind(metrics_summary, data.frame(
      Scenario = sim_name,
      MSE_TTI = round(mse_val, 5),
      GES_Accuracy = round(accuracy_val, 3),
      F1_Score = round(f1_score, 3)
    ))
    
  
    # --- 6. SHAP ANALYSIS ---
    message(paste("Starting SHAP analysis for", sim_name, "..."))
    
    # Helper Matrices for fastshap
    x_train_matrix <- as.matrix(dataTrain[, vari])
    x_test_matrix  <- as.matrix(dataTest[, vari])
    colnames(x_test_matrix) <- vari
    
    # Predict wrapper logic: R matrix -> CUDA Tensor -> Output -> CPU Vector
    predict_fun <- function(object, newdata) {
      batch_tensor <- torch_tensor(as.matrix(newdata), dtype = torch_float(), device = device)
      with_no_grad({
        pred <- as.numeric(object(batch_tensor)$cpu())
      })
      return(pred)
    }
    
    # Background sample (500 is standard to speed up expectations)
    set.seed(123)
    bg_idx <- sample(nrow(x_train_matrix), min(500, nrow(x_train_matrix)))
    
    # Explain sample: We take 2000 points from the test set for the plot.
    # Plotting all 150k+ test points creates massive file sizes and overlapping blobs.
    test_idx <- sample(nrow(x_test_matrix), min(2000, nrow(x_test_matrix)))
    
    e_time <- system.time({
      shap_values <- fastshap::explain(
        object = model,
        X = x_train_matrix[bg_idx, ], 
        newdata = x_test_matrix[test_idx, ],
        nsim = 50,
        pred_wrapper = predict_fun,
        shap_only = FALSE
      )
    })
    
    message(sprintf("SHAP calculated in %.2f seconds.", e_time[3]))
    
    print(paste("Save shap values for :",model_file_name))
    # --- NEW: EXPORT SHAP DATA FOR PLOTTING ---
    shap_export_data <- list(
      shap_vals = shap_values,
      feature_data = x_test_matrix[test_idx, ]
    )
    # 
    export_filename <- paste0("SHAP_data_MLP70SPAT_", sim_name, ".rds")
    saveRDS(shap_export_data, file = export_filename)
    message("SHAP data saved to: ", export_filename)
    
    
    # --- 7. SHAPVIZ PLOTTING ---
    shp <- shapviz(shap_values, X = x_test_matrix[test_idx, ])
    
    # 1. Beeswarm Plot (Feature impact and distribution)
    p_bee <- sv_importance(shp, kind = "beeswarm", max_display = nvari, show_numbers = TRUE) + 
      theme_minimal() +
      theme(
        plot.title = element_text(face = "bold", hjust=0.5, size=16),
        axis.text.y = element_text(size = 12, face = "bold"), 
        axis.title.x = element_text(size = 12, face = "bold"),
        legend.text = element_text(size = 10),
        legend.title = element_text(size = 12)
      ) + 
      labs(title = paste("SHAP Summary (TTI):", sim_name))
    
    # 2. Bar Plot (Mean Absolute Global Importance) with Gradient Colors
    # Extract the raw SHAP values from the fastshap object
    raw_shap_matrix <- as.matrix(shap_values$shapley_values)
    
    # Calculate mean absolute SHAP values (Global Importance)
    global_importance <- colMeans(abs(raw_shap_matrix))
    
    # Create a clean dataframe for ggplot, sorted by importance
    importance_df <- data.frame(
      feature = names(global_importance),
      mean_abs_shap = as.numeric(global_importance)
    ) %>%
      arrange(mean_abs_shap) %>% # Sort ascending so the largest is at the top when flipped
      mutate(feature = factor(feature, levels = feature)) # Lock the factor order for ggplot
    
    # Generate the plot
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
      labs(
        title = paste("Global Feature Importance:", sim_name),
        x = "Mean Absolute SHAP Value (Impact on TTI)",
        y = NULL
      ) +
      expand_limits(x = max(importance_df$mean_abs_shap) * 1.15) # Expands the right edge
   
     # Print to console/viewer
    print(p_bee)
    print(p_bar)
    
    # Save plots automatically
    #ggsave(filename = paste0(model_name,"_SHAP_Beeswarm_", sim_name, ".png"), plot = p_bee, width = 8, height = 6, bg = "white")
    #ggsave(filename = paste0(model_name,"_SHAP_Bar_", sim_name, ".png"), plot = p_bar, width = 8, height = 6, bg = "white")
    

 } ## end


#
# --- 7. FINAL COMPARATIVE OUTPUT ---
print("==================================================")
print("COMPILATION SUMMARY TABLE (MLP70 SPATIAL EXTRAPOLATION):")
print(metrics_summary)

metrics_summary$Model <- model_name

metrics_summary_file <- paste0(model_name,"_Training_Summary_Table",".rds")
saveRDS(metrics_summary, file = metrics_summary_file)
message("Metrics summary saved to: ", metrics_summary_file)

library(gt)

metrics_summary %>% gt

