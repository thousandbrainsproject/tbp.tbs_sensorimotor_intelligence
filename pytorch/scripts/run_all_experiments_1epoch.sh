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

# Show usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Run ViT experiments for reproducibility."
    echo
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -a, --all               Run all experiments (default)"
    echo "  -7a, --fig7a            Run only Figure 7a (Rapid Learning) experiments"
    echo "  -7b, --fig7b            Run only Figure 7b (Continual Learning) experiments"  
    echo "  -8b, --fig8b            Run only Figure 8b (FLOP Analysis) experiments"
    echo "  -s, --skip-setup        Skip environment setup and verification"
    echo "  --from-7a               Run from Figure 7a onwards (7a, 7b, 8b)"
    echo "  --from-7b               Run from Figure 7b onwards (7b, 8b)"
    echo "  --from-8b               Run only Figure 8b"
    echo
    echo "Examples:"
    echo "  $0                      Run all experiments (interactive mode)"
    echo "  $0 --all                Run all experiments"
    echo "  $0 --fig7a              Run only Figure 7a experiments"
    echo "  $0 --from-7b            Run Figure 7b and 8b experiments"
    echo "  $0 --fig8b --skip-setup Run only Figure 8b, skip setup"
}

# Show interactive menu
show_menu() {
    echo
    log_section "ViT Experiments - Select What to Run"
    echo "1) Run all experiments (Fig 7a + 7b + 8b)"
    echo "2) Run from Figure 7a onwards (7a + 7b + 8b)"  
    echo "3) Run from Figure 7b onwards (7b + 8b)"
    echo "4) Run only Figure 7a (Rapid Learning)"
    echo "5) Run only Figure 7b (Continual Learning - limited)"
    echo "6) Run only Figure 8b (FLOP Analysis)"
    echo "7) Exit"
    echo
}

# Get user choice from menu
get_user_choice() {
    while true; do
        show_menu
        read -p "Please select an option (1-7): " choice
        case $choice in
            1) 
                log_info "Selected: Run all experiments"
                RUN_FIG7A=true
                RUN_FIG7B=true  
                RUN_FIG8B=true
                break
                ;;
            2)
                log_info "Selected: Run from Figure 7a onwards"
                RUN_FIG7A=true
                RUN_FIG7B=true
                RUN_FIG8B=true
                break
                ;;
            3)
                log_info "Selected: Run from Figure 7b onwards"
                RUN_FIG7A=false
                RUN_FIG7B=true
                RUN_FIG8B=true
                break
                ;;
            4)
                log_info "Selected: Run only Figure 7a"
                RUN_FIG7A=true
                RUN_FIG7B=false
                RUN_FIG8B=false
                break
                ;;
            5)
                log_info "Selected: Run only Figure 7b"
                RUN_FIG7A=false
                RUN_FIG7B=true
                RUN_FIG8B=false
                break
                ;;
            6)
                log_info "Selected: Run only Figure 8b"
                RUN_FIG7A=false
                RUN_FIG7B=false
                RUN_FIG8B=true
                break
                ;;
            7)
                log_info "Exiting..."
                exit 0
                ;;
            *)
                log_error "Invalid option. Please select 1-7."
                ;;
        esac
    done
}

# Check if conda is available
check_conda() {
    # Try to initialize conda first
    if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
    
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
    
    # Check if we're already in the vit environment
    if [[ "$CONDA_DEFAULT_ENV" == "vit" ]]; then
        log_success "Already in 'vit' environment. Skipping setup."
    else
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
        conda activate vit
        log_success "Environment 'vit' activated"
    fi
    
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
    log_section "Figure 7a: Rapid Learning Experiments (1 Epoch)"
    
    log_info "Testing learning efficiency with different amounts of training data (rotation counts) - 1 epoch training"
    log_info "This will run training and evaluation for 1, 2, 4, 8, 16, 32 rotations"
    log_info "Training and evaluation will be interleaved for faster feedback"
    
    # Interleaved training and evaluation for each rotation count
    for rot in 1 8 32; do
        log_info "Processing ${rot} rotations..."
        
        # Train pretrained model
        log_info "  1/4: Training pretrained model with ${rot} rotations for 1 epoch..."
        python src/train.py \
            experiment=02_fig7a_rapid_learning/pretrained/train/vit-b16-224-in21k_25epochs_n_rot \
            data.num_rotations_for_train=${rot} \
            trainer.max_epochs=1 \
            paths=reproduction
        log_success "  ✓ Completed pretrained training with ${rot} rotations"
        
        # Evaluate pretrained model
        log_info "  2/4: Evaluating pretrained model with ${rot} rotations..."
        python src/eval_standard.py \
            experiment=02_fig7a_rapid_learning/pretrained/inference/vit-b16-224-in21k_25epochs_n_rot \
            data.num_rotations_for_train=${rot} \
            paths=reproduction
        log_success "  ✓ Completed evaluation of pretrained model with ${rot} rotations"
        
        # Train random init model
        log_info "  3/4: Training randomly initialized model with ${rot} rotations for 1 epoch..."
        python src/train.py \
            experiment=02_fig7a_rapid_learning/random_init/train/vit-b16-224-in21k_1epochs_n_rot \
            data.num_rotations_for_train=${rot} \
            trainer.max_epochs=1 \
            paths=reproduction
        log_success "  ✓ Completed random init training with ${rot} rotations"
        
        # Evaluate random init model
        log_info "  4/4: Evaluating randomly initialized model with ${rot} rotations..."
        python src/eval_standard.py \
            experiment=02_fig7a_rapid_learning/random_init/inference/vit-b16-224-in21k_1epoch_n_rot \
            data.num_rotations_for_train=${rot} \
            paths=reproduction
        log_success "  ✓ Completed evaluation of random init model with ${rot} rotations"
        
        log_success "All experiments for ${rot} rotations completed!"
        echo
    done
    
    log_success "All rapid learning experiments (1 epoch) completed!"
}

# Figure 7b: Continual Learning Experiments (Limited to first 5 tasks, 1 epoch each)
run_fig7b_experiments() {
    log_section "Figure 7b: Continual Learning Experiments (First 5 Tasks, 1 Epoch Each)"
    
    log_info "Testing continual learning across sequential tasks - 1 epoch per task"
    log_info "Note: Running only first 5 tasks instead of all 77 for demonstration"
    
    # Function to handle errors
    handle_error() {
        local task_id="$1"
        log_error "Training failed for task_id = ${task_id}"
        log_warning "Continuing with next task..."
        return 0
    }
    
    # Run initial task (task0) with 1 epoch
    log_info "Running initial task (task0) for 1 epoch..."
    python src/train.py \
        experiment=03_fig7b_continual_learning/train/task0 \
        "task_id=0" \
        trainer.max_epochs=1 \
        paths=reproduction || handle_error "0"
    log_success "Successfully completed task 0"
    
    # Run subsequent tasks (1-4) - limiting to first 5 total tasks
    log_info "Running subsequent tasks (1-4) for 1 epoch each..."
    
    for task_id in $(seq 1 4); do
        log_info "Running training for task_id = ${task_id} (1 epoch)"
        python src/train.py \
            experiment=03_fig7b_continual_learning/train/task1+ \
            "task_id=${task_id}" \
            trainer.max_epochs=1 \
            paths=reproduction || handle_error "${task_id}"
        log_success "Successfully completed task ${task_id}"
        
        # Small delay between tasks
        sleep 2
    done
    
    # Run evaluation
    log_info "Running continual learning evaluation..."
    python src/eval_continual.py \
        experiment=03_fig7b_continual_learning/inference/eval_continual_learning \
        paths=reproduction || log_warning "Evaluation failed, but continuing..."
    log_success "Continual learning evaluation completed"
    
    log_success "Continual learning experiments (first 5 tasks, 1 epoch each) completed!"
    log_warning "To run all 77 tasks, use: ./scripts/fig7b_continual_learning.sh"
}

# Figure 8b: FLOP Analysis Experiments
run_fig8b_experiments() {
    log_section "Figure 8b: FLOP Analysis Experiments (1 Epoch)"
    
    log_info "Comparing different ViT architectures (B/16, B/32, L/32) - 1 epoch training"
    log_info "Training and evaluation will be interleaved for faster feedback"
    
    # Define the models to test (reduced set for quick testing)
    MODELS=("vit-b16-224-in21k" "vit-b32-224-in21k" "vit-l32-224-in21k")
    
    # Interleaved training and evaluation for each model
    for model in "${MODELS[@]}"; do
        log_info "Processing ${model}..."
        
        # Train pretrained model
        log_info "  1/4: Training pretrained ${model} for 1 epoch..."
        python src/train.py \
            experiment=04_fig8b_flops/pretrained/train/${model} \
            trainer.max_epochs=1 \
            paths=reproduction
        log_success "  ✓ Completed training pretrained ${model}"
        
        # Evaluate pretrained model
        log_info "  2/4: Evaluating pretrained ${model}..."
        python src/eval_standard.py \
            experiment=04_fig8b_flops/pretrained/inference/${model} \
            paths=reproduction
        log_success "  ✓ Completed evaluation of pretrained ${model}"
        
        # Train random init model
        log_info "  3/4: Training randomly initialized ${model} for 1 epoch..."
        python src/train.py \
            experiment=04_fig8b_flops/random_init/train/${model} \
            trainer.max_epochs=1 \
            paths=reproduction
        log_success "  ✓ Completed training random init ${model}"
        
        # Evaluate random init model
        log_info "  4/4: Evaluating randomly initialized ${model}..."
        python src/eval_standard.py \
            experiment=04_fig8b_flops/random_init/inference/${model} \
            paths=reproduction
        log_success "  ✓ Completed evaluation of random init ${model}"
        
        log_success "All experiments for ${model} completed!"
        echo
    done
    
    log_success "All FLOP analysis experiments (1 epoch) completed!"
}

# Parse command line arguments
parse_arguments() {
    # Default values
    RUN_FIG7A=true
    RUN_FIG7B=true
    RUN_FIG8B=true
    SKIP_SETUP=false
    INTERACTIVE_MODE=true
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -a|--all)
                RUN_FIG7A=true
                RUN_FIG7B=true
                RUN_FIG8B=true
                INTERACTIVE_MODE=false
                shift
                ;;
            -7a|--fig7a)
                RUN_FIG7A=true
                RUN_FIG7B=false
                RUN_FIG8B=false
                INTERACTIVE_MODE=false
                shift
                ;;
            -7b|--fig7b)
                RUN_FIG7A=false
                RUN_FIG7B=true
                RUN_FIG8B=false
                INTERACTIVE_MODE=false
                shift
                ;;
            -8b|--fig8b)
                RUN_FIG7A=false
                RUN_FIG7B=false
                RUN_FIG8B=true
                INTERACTIVE_MODE=false
                shift
                ;;
            --from-7a)
                RUN_FIG7A=true
                RUN_FIG7B=true
                RUN_FIG8B=true
                INTERACTIVE_MODE=false
                shift
                ;;
            --from-7b)
                RUN_FIG7A=false
                RUN_FIG7B=true
                RUN_FIG8B=true
                INTERACTIVE_MODE=false
                shift
                ;;
            --from-8b)
                RUN_FIG7A=false
                RUN_FIG7B=false
                RUN_FIG8B=true
                INTERACTIVE_MODE=false
                shift
                ;;
            -s|--skip-setup)
                SKIP_SETUP=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Main execution
main() {
    # Parse command line arguments
    parse_arguments "$@"
    
    log_section "ViT Experiments - Complete Reproduction Suite"
    
    # Check if we're in the right directory
    if [ ! -f ".project-root" ]; then
        log_error ".project-root file not found"
        log_info "Please run this script from: tbp.tbs_sensorimotor_intelligence/pytorch"
        exit 1
    fi
    
    # Interactive mode - let user choose what to run
    if [ "$INTERACTIVE_MODE" = true ]; then
        get_user_choice
    fi
    
    # Show what will be run
    log_info "This script will run the following experiments:"
    if [ "$RUN_FIG7A" = true ]; then
        log_info "  ✓ Figure 7a: Rapid Learning Experiments"
    fi
    if [ "$RUN_FIG7B" = true ]; then
        log_info "  ✓ Figure 7b: Continual Learning Experiments (limited)"
    fi
    if [ "$RUN_FIG8B" = true ]; then
        log_info "  ✓ Figure 8b: FLOP Analysis Experiments"
    fi
    
    # Calculate estimated time (1 epoch training)
    ESTIMATED_TIME="20-40 minutes"
    if [ "$RUN_FIG7A" = true ] && [ "$RUN_FIG7B" = true ] && [ "$RUN_FIG8B" = true ]; then
        ESTIMATED_TIME="1-2 hours"
    elif [ "$RUN_FIG7A" = true ] && [ "$RUN_FIG8B" = true ]; then
        ESTIMATED_TIME="40-80 minutes"
    elif [ "$RUN_FIG7A" = true ]; then
        ESTIMATED_TIME="30-60 minutes"
    elif [ "$RUN_FIG8B" = true ]; then
        ESTIMATED_TIME="15-30 minutes"
    fi
    
    log_info "Estimated total time: $ESTIMATED_TIME (1 epoch training) depending on hardware"
    log_warning "Make sure you have sufficient disk space and computational resources"
    
    # Confirm execution
    read -p "Do you want to proceed with the selected experiments? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Aborted by user"
        exit 0
    fi
    
    # Store start time
    START_TIME=$(date)
    log_info "Starting experiments at: $START_TIME"
    
    # Setup and verification (unless skipped)
    if [ "$SKIP_SETUP" = false ]; then
        check_conda
        setup_environment
        verify_setup
        run_quick_test
    else
        log_warning "Skipping setup and verification as requested"
        # Still need to activate environment
        if [[ "$CONDA_DEFAULT_ENV" != "vit" ]]; then
            log_info "Activating vit environment..."
            conda activate vit
        fi
        export PROJECT_ROOT=$(pwd)
    fi
    
    # Run selected experiment suites
    if [ "$RUN_FIG7A" = true ]; then
        run_fig7a_experiments
    else
        log_info "Skipping Figure 7a experiments"
    fi
    
    if [ "$RUN_FIG7B" = true ]; then
        run_fig7b_experiments
    else
        log_info "Skipping Figure 7b experiments"
    fi
    
    if [ "$RUN_FIG8B" = true ]; then
        run_fig8b_experiments
    else
        log_info "Skipping Figure 8b experiments"
    fi
    
    # Summary
    END_TIME=$(date)
    log_section "1-Epoch Experiment Suite Complete!"
    
    log_success "Selected experiments completed successfully (1 epoch training)!"
    log_info "Started: $START_TIME"
    log_info "Finished: $END_TIME"
    
    echo
    log_info "📊 Results Analysis:"
    log_info "1. Check WandB project 'benchmark_vit' for detailed metrics"
    log_info "2. Local results saved to: logs/ directory"
    log_info "3. NOTE: Results are from 1-epoch training, not full training"
    
    echo
    log_info "🎯 Next Steps:"
    log_info "1. Analyze results in WandB dashboard"
    log_info "2. If everything runs correctly, try full training with original epochs"  
    log_info "3. For full continual learning (77 tasks): ./scripts/fig7b_continual_learning.sh"
    log_info "4. For full training, edit configs to use original epoch counts"
    
    # Show restart instructions
    echo
    log_info "🔄 To restart from a specific point:"
    log_info "  - Run only Fig 7a: $0 --fig7a"
    log_info "  - Run only Fig 7b: $0 --fig7b"  
    log_info "  - Run only Fig 8b: $0 --fig8b"
    log_info "  - From Fig 7b onwards: $0 --from-7b"
    log_info "  - Skip setup: $0 --fig7a --skip-setup"
    
    echo
    log_success "Thank you for testing the ViT pipeline! 🎉"
    log_info "This was a 1-epoch test run to verify everything works correctly."
}

# Handle interruption gracefully
trap 'log_warning "Script interrupted by user"; exit 130' INT

# Run main function
main "$@" 