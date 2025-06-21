#!/bin/bash

# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Comprehensive script to set up environment and run all ViT experiments
# This script will:
# 1. Set up conda environment (if needed)
# 2. Verify setup
# 3. Run all three experiment suites (Fig 7a, 7b limited, 8b)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_section() {
    echo
    echo "================================================================================"
    echo -e "${BLUE}🚀 $1${NC}"
    echo "================================================================================"
}

# Check if conda is available
check_conda() {
    if ! command -v conda &> /dev/null; then
        log_error "Conda not found. Please install Miniconda or Anaconda first."
        echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    log_success "Conda found: $(conda --version)"
}

# Setup environment
setup_environment() {
    log_section "Setting up Conda Environment"
    
    # Check if environment already exists
    if conda env list | grep -q "^vit "; then
        log_warning "Environment 'vit' already exists. Skipping creation."
        log_info "To recreate environment, run: conda env remove -n vit"
    else
        log_info "Creating conda environment from environment.yaml..."
        conda env create -f environment.yaml
        log_success "Environment 'vit' created successfully"
    fi
    
    # Activate environment
    log_info "Activating environment..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate vit
    log_success "Environment 'vit' activated"
    
    # Set PROJECT_ROOT
    export PROJECT_ROOT=$(pwd)
    log_info "PROJECT_ROOT set to: $PROJECT_ROOT"
}

# Verify setup
verify_setup() {
    log_section "Verifying Setup"
    
    log_info "Running setup verification script..."
    python scripts/verify_setup.py
    
    if [ $? -eq 0 ]; then
        log_success "Setup verification passed"
    else
        log_error "Setup verification failed. Please fix issues above."
        exit 1
    fi
}

# Run a quick test
run_quick_test() {
    log_section "Running Quick Test"
    
    log_info "Running end-to-end test..."
    ./scripts/quick_test.sh
    
    if [ $? -eq 0 ]; then
        log_success "Quick test passed"
    else
        log_warning "Quick test had issues, but proceeding with experiments"
    fi
}

# Figure 7a: Rapid Learning Experiments
run_fig7a_experiments() {
    log_section "Figure 7a: Rapid Learning Experiments"
    
    log_info "Testing learning efficiency with different amounts of training data (rotation counts)"
    log_info "This will run training and evaluation for 1, 2, 4, 8, 16, 32 rotations"
    
    # Section 1: Training with pretrained initialization
    log_info "Section 1: Training with pretrained initialization (25 epochs)..."
    
    for rot in 1 2 4 8 16 32; do
        log_info "Training pretrained model with ${rot} rotations for 25 epochs..."
        python src/train.py \
            experiment=02_fig7a_rapid_learning/pretrained/train/vit-b16-224-in21k_25epochs_n_rot \
            data.num_rotations_for_train=${rot}
        log_success "Completed pretrained training with ${rot} rotations"
    done
    
    # Section 2: Training with random initialization (75 epochs)
    log_info "Section 2: Training with random initialization (75 epochs)..."
    
    for rot in 1 2 4 8 16 32; do
        log_info "Training randomly initialized model with ${rot} rotations for 75 epochs..."
        python src/train.py \
            experiment=02_fig7a_rapid_learning/random_init/train/vit-b16-224-in21k_75epochs_n_rot \
            data.num_rotations_for_train=${rot}
        log_success "Completed random init training (75 epochs) with ${rot} rotations"
    done
    
    # Section 3: Training with random initialization (1 epoch)
    log_info "Section 3: Training with random initialization (1 epoch)..."
    
    for rot in 1 2 4 8 16 32; do
        log_info "Training randomly initialized model with ${rot} rotations for 1 epoch..."
        python src/train.py \
            experiment=02_fig7a_rapid_learning/random_init/train/vit-b16-224-in21k_1epochs_n_rot \
            data.num_rotations_for_train=${rot}
        log_success "Completed random init training (1 epoch) with ${rot} rotations"
    done
    
    # Section 4: Evaluation of pretrained models
    log_info "Section 4: Evaluating pretrained models (25 epochs)..."
    
    for rot in 1 2 4 8 16 32; do
        log_info "Testing pretrained model with ${rot} rotations (25 epochs)..."
        python src/eval.py \
            experiment=02_fig7a_rapid_learning/pretrained/inference/vit-b16-224-in21k_25epochs_n_rot \
            data.num_rotations_for_train=${rot}
        log_success "Completed evaluation of pretrained model with ${rot} rotations"
    done
    
    # Section 5: Evaluation of randomly initialized models (75 epochs)
    log_info "Section 5: Evaluating randomly initialized models (75 epochs)..."
    
    for rot in 1 2 4 8 16 32; do
        log_info "Testing randomly initialized model with ${rot} rotations (75 epochs)..."
        python src/eval.py \
            experiment=02_fig7a_rapid_learning/random_init/inference/vit-b16-224-in21k_75epochs_n_rot \
            data.num_rotations_for_train=${rot}
        log_success "Completed evaluation of random init model (75 epochs) with ${rot} rotations"
    done
    
    # Section 6: Evaluation of randomly initialized models (1 epoch)
    log_info "Section 6: Evaluating randomly initialized models (1 epoch)..."
    
    for rot in 1 2 4 8 16 32; do
        log_info "Testing randomly initialized model with ${rot} rotations (1 epoch)..."
        python src/eval.py \
            experiment=02_fig7a_rapid_learning/random_init/inference/vit-b16-224-in21k_1epoch_n_rot \
            data.num_rotations_for_train=${rot}
        log_success "Completed evaluation of random init model (1 epoch) with ${rot} rotations"
    done
    
    log_success "All rapid learning experiments completed!"
}

# Figure 7b: Continual Learning Experiments (Limited to first 5 tasks)
run_fig7b_experiments() {
    log_section "Figure 7b: Continual Learning Experiments (First 5 Tasks)"
    
    log_info "Testing continual learning across sequential tasks"
    log_info "Note: Running only first 5 tasks instead of all 77 for demonstration"
    
    # Function to handle errors
    handle_error() {
        local task_id="$1"
        log_error "Training failed for task_id = ${task_id}"
        echo "Please check the logs for more details"
        exit 1
    }
    
    # Run initial task (task0)
    log_info "Running initial task (task0)..."
    python src/train.py experiment=03_fig7b_continual_learning/train/task0 "task_id=0" || handle_error "0"
    log_success "Successfully completed task 0"
    
    # Run subsequent tasks (1-4) - limiting to first 5 total tasks
    log_info "Running subsequent tasks (1-4)..."
    
    for task_id in $(seq 1 4); do
        log_info "Running training for task_id = ${task_id}"
        python src/train.py experiment=03_fig7b_continual_learning/train/task1+ "task_id=${task_id}" || handle_error "${task_id}"
        log_success "Successfully completed task ${task_id}"
        
        # Small delay between tasks
        sleep 2
    done
    
    # Run evaluation
    log_info "Running continual learning evaluation..."
    python src/eval_continual.py experiment=03_fig7b_continual_learning/inference/eval_continual_learning
    log_success "Continual learning evaluation completed"
    
    log_success "Continual learning experiments (first 5 tasks) completed!"
    log_warning "To run all 77 tasks, use: ./scripts/fig7b_continual_learning.sh"
}

# Figure 8b: FLOP Analysis Experiments
run_fig8b_experiments() {
    log_section "Figure 8b: FLOP Analysis Experiments"
    
    log_info "Comparing different ViT architectures (B/16, B/32, L/16, L/32, H/14)"
    
    # Define the models to test
    MODELS=("vit-b16-224-in21k" "vit-b32-224-in21k" "vit-h14-224-in21k" "vit-l16-224-in21k" "vit-l32-224-in21k")
    
    # Section 1: Training with pretrained initialization
    log_info "Section 1: Training with pretrained initialization..."
    
    for model in "${MODELS[@]}"; do
        log_info "Training pretrained ${model}..."
        python src/train.py experiment=04_fig8b_flops/pretrained/train/${model}
        log_success "Completed training pretrained ${model}"
    done
    
    # Section 2: Training with random initialization
    log_info "Section 2: Training with random initialization..."
    
    for model in "${MODELS[@]}"; do
        log_info "Training randomly initialized ${model}..."
        python src/train.py experiment=04_fig8b_flops/random_init/train/${model}
        log_success "Completed training random init ${model}"
    done
    
    # Section 3: Evaluation of pretrained models
    log_info "Section 3: Evaluating pretrained models..."
    
    for model in "${MODELS[@]}"; do
        log_info "Testing pretrained ${model}..."
        python src/eval.py experiment=04_fig8b_flops/pretrained/inference/${model}
        log_success "Completed evaluation of pretrained ${model}"
    done
    
    # Section 4: Evaluation of randomly initialized models
    log_info "Section 4: Evaluating randomly initialized models..."
    
    for model in "${MODELS[@]}"; do
        log_info "Testing randomly initialized ${model}..."
        python src/eval.py experiment=04_fig8b_flops/random_init/inference/${model}
        log_success "Completed evaluation of random init ${model}"
    done
    
    log_success "All FLOP analysis experiments completed!"
}

# Main execution
main() {
    log_section "ViT Experiments - Complete Reproduction Suite"
    
    log_info "This script will run all ViT experiments for reproducibility"
    log_info "Estimated total time: 4-8 hours depending on hardware"
    log_warning "Make sure you have sufficient disk space and computational resources"
    
    # Confirm execution
    read -p "Do you want to proceed with all experiments? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Aborted by user"
        exit 0
    fi
    
    # Check if we're in the right directory
    if [ ! -f ".project-root" ]; then
        log_error ".project-root file not found"
        log_info "Please run this script from: tbp.tbs_sensorimotor_intelligence/pytorch"
        exit 1
    fi
    
    # Store start time
    START_TIME=$(date)
    log_info "Starting experiments at: $START_TIME"
    
    # Setup and verification
    check_conda
    setup_environment
    verify_setup
    run_quick_test
    
    # Run all experiment suites
    run_fig7a_experiments
    run_fig7b_experiments
    run_fig8b_experiments
    
    # Summary
    END_TIME=$(date)
    log_section "Experiment Suite Complete!"
    
    log_success "All experiments completed successfully!"
    log_info "Started: $START_TIME"
    log_info "Finished: $END_TIME"
    
    echo
    log_info "📊 Results Analysis:"
    log_info "1. Check WandB project 'benchmark_vit' for detailed metrics"
    log_info "2. Local results saved to: ~/tbp/results/dmc/results/vit/logs/"
    log_info "3. Compare with expected results in REPRODUCE_RESULTS.md"
    
    echo
    log_info "🎯 Next Steps:"
    log_info "1. Analyze results in WandB dashboard"
    log_info "2. Compare your results with published benchmarks"  
    log_info "3. For full continual learning (77 tasks): ./scripts/fig7b_continual_learning.sh"
    
    echo
    log_success "Thank you for reproducing our ViT results! 🎉"
}

# Handle interruption gracefully
trap 'log_warning "Script interrupted by user"; exit 130' INT

# Run main function
main "$@" 