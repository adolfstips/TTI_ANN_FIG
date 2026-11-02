# ==============================================================================
# Script: 00_Run_All_Workflow.R
# Title: Master End-to-End Execution Pipeline (Raw Data -> Models -> Tables/Figures)
# Description: Sequentially executes the complete computational workflow for
#              Stips et al. (2026). Supports full GPU model training or fast
#              table/figure regeneration from pre-computed ./Output/*.rds files.
# Reference: Stips et al. (2026) - Environmental Modelling & Software
# License: GPL-3.0
# ==============================================================================

# --- 1. WORKFLOW CONFIGURATION ---
# Set RUN_TRAINING = TRUE to retrain all neural networks from scratch (GPU recommended).
# Set RUN_TRAINING = FALSE to skip training and regenerate Tables & Figures from ./Output/*.rds.
RUN_TRAINING <- TRUE

# Ensure output directories exist
for (d in c("./Output", "./Tables", "./Figures")) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
}

# Automatically unzip the NetCDF dataset if only the .zip archive is present
nc_file  <- "./GES_Baltic_5year_ALLE_Combined_4D.nc"
zip_file <- "./GES_Baltic_5year_ALLE_Combined_4D.nc.zip"

if (!file.exists(nc_file)) {
  if (file.exists(zip_file)) {
    message("Unzipping input dataset: ", zip_file)
    unzip(zip_file, exdir = "./")
  } else {
    stop("Input dataset not found! Please place ", nc_file, " or ", zip_file, " in the root directory.")
  }
}

# Helper function to run each script in a clean isolated environment with timing & GPU cleanup
run_step <- function(script_name) {
  if (!file.exists(script_name)) stop("Script not found: ", script_name)
  
  cat("\n======================================================================\n")
  cat("EXECUTING:", script_name, "| Started at:", format(Sys.time(), "%H:%M:%S"), "\n")
  cat("======================================================================\n")
  
  t_start <- Sys.time()
  step_env <- new.env(parent = globalenv())
  source(script_name, local = step_env, echo = FALSE)
  
  rm(step_env)
  gc()
  if (requireNamespace("torch", quietly = TRUE) && torch::cuda_is_available()) {
    torch::cuda_empty_cache()
  }
  
  elapsed <- round(as.numeric(difftime(Sys.time(), t_start, units = "mins")), 2)
  cat("COMPLETED:", script_name, "in", elapsed, "minutes.\n")
}

pipeline_start <- Sys.time()

# --- 2. PHASE 1: NEURAL NETWORK TRAINING & SHAP EXTRACTION (SCRIPTS 01-05) ---
if (RUN_TRAINING) {
  training_scripts <- c(
    "01_Kfold_MLP_SPAT.R",
    "02_Kfold_LSTM_SPAT.R",
    "03_Rolling_MLP_TIME.R",
    "04_Rolling_LSTM_TIME.R",
    "05_Spatial_Ensemble_MLP_SPAT.R"
  )
  
  for (scr in training_scripts) {
    run_step(scr)
  }
} else {
  message("\nSkipping Phase 1 (Model Training). Using pre-computed .rds files in ./Output/")
}

# --- 3. PHASE 2: TABLE & FIGURE GENERATION (SCRIPTS 06-10) ---
postproc_scripts <- c(
  "06_Generate_Manuscript_Tables.R",
  "07_Plot_Fig1_Spatial_Block_Map.R",
  "08_Plot_Figs2_3_4_SHAP_Analysis.R",
  "09_Plot_Figs5_6_S3_Spatial_Ensemble.R",
  "10_Plot_FigsS1_S2_Confusion_Matrices.R"
)

for (scr in postproc_scripts) {
  run_step(scr)
}

total_time <- round(as.numeric(difftime(Sys.time(), pipeline_start, units = "mins")), 2)
cat("\n======================================================================\n")
cat("ALL WORKFLOW STEPS COMPLETED SUCCESSFULLY IN", total_time, "MINUTES!\n")
cat("  - Intermediate RDS Data : ./Output/\n")
cat("  - Manuscript Tables     : ./Tables/\n")
cat("  - Publication Figures   : ./Figures/\n")
cat("======================================================================\n")

