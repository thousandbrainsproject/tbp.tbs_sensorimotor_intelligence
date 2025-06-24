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
    echo "1) Run all experiments (Fig 7a + 7b + 8b - Training & Inference)"
    echo "2) Figure 7a (Rapid Learning) - Training only"
    echo "3) Figure 7a (Rapid Learning) - Inference only"
    echo "4) Figure 7b (Continual Learning) - Training only"
    echo "5) Figure 7b (Continual Learning) - Inference only"
    echo "6) Figure 8b (FLOP Analysis) - Training only"
    echo "7) Figure 8b (FLOP Analysis) - Inference only"
    echo "8) Exit"
    echo
}

# Get user choice from menu
get_user_choice() {
    while true; do
        show_menu
        read -p "Please select an option (1-8): " choice
        case $choice in
            1) 
                log_info "Selected: Run all experiments (Training & Inference)"
                RUN_FIG7A=true
                RUN_FIG7B=true  
                RUN_FIG8B=true
                RUN_FIG7A_TRAINING=true
                RUN_FIG7A_INFERENCE=true
                RUN_FIG7B_TRAINING=true
                RUN_FIG7B_INFERENCE=true
                RUN_FIG8B_TRAINING=true
                RUN_FIG8B_INFERENCE=true
                break
                ;;
            2)
                log_info "Selected: Figure 7a (Rapid Learning) - Training only"
                RUN_FIG7A=true
                RUN_FIG7B=false
                RUN_FIG8B=false
                RUN_FIG7A_TRAINING=true
                RUN_FIG7A_INFERENCE=false
                break
                ;;
            3)
                log_info "Selected: Figure 7a (Rapid Learning) - Inference only"
                RUN_FIG7A=true
                RUN_FIG7B=false
                RUN_FIG8B=false
                RUN_FIG7A_TRAINING=false
                RUN_FIG7A_INFERENCE=true
                break
                ;;
            4)
                log_info "Selected: Figure 7b (Continual Learning) - Training only"
                RUN_FIG7A=false
                RUN_FIG7B=true
                RUN_FIG8B=false
                RUN_FIG7B_TRAINING=true
                RUN_FIG7B_INFERENCE=false
                break
                ;;
            5)
                log_info "Selected: Figure 7b (Continual Learning) - Inference only"
                RUN_FIG7A=false
                RUN_FIG7B=true
                RUN_FIG8B=false
                RUN_FIG7B_TRAINING=false
                RUN_FIG7B_INFERENCE=true
                break
                ;;
            6)
                log_info "Selected: Figure 8b (FLOP Analysis) - Training only"
                RUN_FIG7A=false
                RUN_FIG7B=false
                RUN_FIG8B=true
                RUN_FIG8B_TRAINING=true
                RUN_FIG8B_INFERENCE=false
                break
                ;;
            7)
                log_info "Selected: Figure 8b (FLOP Analysis) - Inference only"
                RUN_FIG7A=false
                RUN_FIG7B=false
                RUN_FIG8B=true
                RUN_FIG8B_TRAINING=false
                RUN_FIG8B_INFERENCE=true
                break
                ;;
            8)
                log_info "Exiting..."
                exit 0
                ;;
            *)
                log_error "Invalid option. Please select 1-8."
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

# Figure 7a: Rapid Learning Training
run_fig7a_training() {
    log_section "Figure 7a: Rapid Learning Training"
    
    log_info "Training models with different amounts of training data (rotation counts)"
    log_info "This will train models for 1, 2, 4, 8, 16, 32 rotations"
    
    # Section 1: Training with pretrained initialization
    log_info "Section 1: Training with pretrained initialization (25 epochs)..."
    
    for rot in 1 2 4 8 16 32; do
        log_info "Training pretrained model with ${rot} rotations for 25 epochs..."
        python src/train.py \
            experiment=02_fig7a_rapid_learning/pretrained/train/vit-b16-224-in21k_25epochs_n_rot \
            data.num_rotations_for_train=${rot} \
            paths=reproduction
        log_success "Completed pretrained training with ${rot} rotations"
    done
    
    # Section 2: Training with random initialization (75 epochs)
    log_info "Section 2: Training with random initialization (75 epochs)..."
    
    for rot in 1 2 4 8 16 32; do
        log_info "Training randomly initialized model with ${rot} rotations for 75 epochs..."
        python src/train.py \
            experiment=02_fig7a_rapid_learning/random_init/train/vit-b16-224-in21k_75epochs_n_rot \
            data.num_rotations_for_train=${rot} \
            paths=reproduction
        log_success "Completed random init training (75 epochs) with ${rot} rotations"
    done
    
    # Section 3: Training with random initialization (1 epoch)
    log_info "Section 3: Training with random initialization (1 epoch)..."
    
    for rot in 1 2 4 8 16 32; do
        log_info "Training randomly initialized model with ${rot} rotations for 1 epoch..."
        python src/train.py \
            experiment=02_fig7a_rapid_learning/random_init/train/vit-b16-224-in21k_1epochs_n_rot \
            data.num_rotations_for_train=${rot} \
            paths=reproduction
        log_success "Completed random init training (1 epoch) with ${rot} rotations"
    done
    
    log_success "Figure 7a training completed!"
}

# Figure 7a: Rapid Learning Inference
run_fig7a_inference() {
    log_section "Figure 7a: Rapid Learning Inference"
    
    log_info "Evaluating trained models with different amounts of training data"
    log_info "This will evaluate models trained with 1, 2, 4, 8, 16, 32 rotations"
    
    # Section 1: Evaluation of pretrained models
    log_info "Section 1: Evaluating pretrained models (25 epochs)..."
    
    for rot in 1 2 4 8 16 32; do
        log_info "Testing pretrained model with ${rot} rotations (25 epochs)..."
        python src/eval_standard.py \
            experiment=02_fig7a_rapid_learning/pretrained/inference/vit-b16-224-in21k_25epochs_n_rot \
            data.num_rotations_for_train=${rot} \
            paths=reproduction
        log_success "Completed evaluation of pretrained model with ${rot} rotations"
    done
    
    # Section 2: Evaluation of randomly initialized models (75 epochs)
    log_info "Section 2: Evaluating randomly initialized models (75 epochs)..."
    
    for rot in 1 2 4 8 16 32; do
        log_info "Testing randomly initialized model with ${rot} rotations (75 epochs)..."
        python src/eval_standard.py \
            experiment=02_fig7a_rapid_learning/random_init/inference/vit-b16-224-in21k_75epochs_n_rot \
            data.num_rotations_for_train=${rot} \
            paths=reproduction
        log_success "Completed evaluation of random init model (75 epochs) with ${rot} rotations"
    done
    
    # Section 3: Evaluation of randomly initialized models (1 epoch)
    log_info "Section 3: Evaluating randomly initialized models (1 epoch)..."
    
    for rot in 1 2 4 8 16 32; do
        log_info "Testing randomly initialized model with ${rot} rotations (1 epoch)..."
        python src/eval_standard.py \
            experiment=02_fig7a_rapid_learning/random_init/inference/vit-b16-224-in21k_1epoch_n_rot \
            data.num_rotations_for_train=${rot} \
            paths=reproduction
        log_success "Completed evaluation of random init model (1 epoch) with ${rot} rotations"
    done
    
    log_success "Figure 7a inference completed!"
}

# Figure 7b: Continual Learning Training (Limited to first 5 tasks)
run_fig7b_training() {
    log_section "Figure 7b: Continual Learning Training (First 5 Tasks)"
    
    log_info "Training continual learning across sequential tasks"
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
    python src/train.py experiment=03_fig7b_continual_learning/train/task0 "task_id=0" paths=reproduction || handle_error "0"
    log_success "Successfully completed task 0"
    
    # Run subsequent tasks (1-4) - limiting to first 5 total tasks
    log_info "Running subsequent tasks (1-4)..."
    
    for task_id in $(seq 1 4); do
        log_info "Running training for task_id = ${task_id}"
        python src/train.py experiment=03_fig7b_continual_learning/train/task1+ "task_id=${task_id}" paths=reproduction || handle_error "${task_id}"
        log_success "Successfully completed task ${task_id}"
        
        # Small delay between tasks
        sleep 2
    done
    
    log_success "Figure 7b training completed!"
    log_warning "To run all 77 tasks, use: ./scripts/fig7b_continual_learning.sh"
}

# Figure 7b: Continual Learning Inference
run_fig7b_inference() {
    log_section "Figure 7b: Continual Learning Inference"
    
    log_info "Evaluating continual learning performance"
    log_info "This will evaluate the model trained on the first 5 tasks"
    
    # Run evaluation
    log_info "Running continual learning evaluation..."
    python src/eval_continual.py experiment=03_fig7b_continual_learning/inference/eval_continual_learning paths=reproduction
    log_success "Continual learning evaluation completed"
    
    log_success "Figure 7b inference completed!"
}

# Figure 8b: FLOP Analysis Training
run_fig8b_training() {
    log_section "Figure 8b: FLOP Analysis Training"
    
    log_info "Training different ViT architectures (B/16, B/32, L/16, L/32)"
    
    # Define the models to test
    MODELS=("vit-b16-224-in21k" "vit-b32-224-in21k" "vit-l16-224-in21k" "vit-l32-224-in21k")
    
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
    
    log_success "Figure 8b training completed!"
}

# Figure 8b: FLOP Analysis Inference
run_fig8b_inference() {
    log_section "Figure 8b: FLOP Analysis Inference"
    
    log_info "Evaluating different ViT architectures (B/16, B/32, L/16, L/32)"
    
    # Define the models to test
    MODELS=("vit-b16-224-in21k" "vit-b32-224-in21k" "vit-l16-224-in21k" "vit-l32-224-in21k")
    
    # Section 1: Evaluation of pretrained models
    log_info "Section 1: Evaluating pretrained models..."
    
    for model in "${MODELS[@]}"; do
        log_info "Testing pretrained ${model}..."
        python src/eval_standard.py experiment=04_fig8b_flops/pretrained/inference/${model}
        log_success "Completed evaluation of pretrained ${model}"
    done
    
    # Section 2: Evaluation of randomly initialized models
    log_info "Section 2: Evaluating randomly initialized models..."
    
    for model in "${MODELS[@]}"; do
        log_info "Testing randomly initialized ${model}..."
        python src/eval_standard.py experiment=04_fig8b_flops/random_init/inference/${model}
        log_success "Completed evaluation of random init ${model}"
    done
    
    log_success "Figure 8b inference completed!"
}

# Parse command line arguments
parse_arguments() {
    # Default values
    RUN_FIG7A=true
    RUN_FIG7B=true
    RUN_FIG8B=true
    RUN_FIG7A_TRAINING=true
    RUN_FIG7A_INFERENCE=true
    RUN_FIG7B_TRAINING=true
    RUN_FIG7B_INFERENCE=true
    RUN_FIG8B_TRAINING=true
    RUN_FIG8B_INFERENCE=true
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
        if [ "${RUN_FIG7A_TRAINING:-true}" = true ] && [ "${RUN_FIG7A_INFERENCE:-true}" = true ]; then
            log_info "  ✓ Figure 7a: Rapid Learning Experiments (Training & Inference)"
        elif [ "${RUN_FIG7A_TRAINING:-true}" = true ]; then
            log_info "  ✓ Figure 7a: Rapid Learning Experiments (Training only)"
        else
            log_info "  ✓ Figure 7a: Rapid Learning Experiments (Inference only)"
        fi
    fi
    if [ "$RUN_FIG7B" = true ]; then
        if [ "${RUN_FIG7B_TRAINING:-true}" = true ] && [ "${RUN_FIG7B_INFERENCE:-true}" = true ]; then
            log_info "  ✓ Figure 7b: Continual Learning Experiments (Training & Inference)"
        elif [ "${RUN_FIG7B_TRAINING:-true}" = true ]; then
            log_info "  ✓ Figure 7b: Continual Learning Experiments (Training only)"
        else
            log_info "  ✓ Figure 7b: Continual Learning Experiments (Inference only)"
        fi
    fi
    if [ "$RUN_FIG8B" = true ]; then
        if [ "${RUN_FIG8B_TRAINING:-true}" = true ] && [ "${RUN_FIG8B_INFERENCE:-true}" = true ]; then
            log_info "  ✓ Figure 8b: FLOP Analysis Experiments (Training & Inference)"
        elif [ "${RUN_FIG8B_TRAINING:-true}" = true ]; then
            log_info "  ✓ Figure 8b: FLOP Analysis Experiments (Training only)"
        else
            log_info "  ✓ Figure 8b: FLOP Analysis Experiments (Inference only)"
        fi
    fi
    
    # Calculate estimated time
    ESTIMATED_TIME="1-2 hours"
    if [ "$RUN_FIG7A" = true ] && [ "$RUN_FIG7B" = true ] && [ "$RUN_FIG8B" = true ]; then
        ESTIMATED_TIME="4-8 hours"
    elif [ "$RUN_FIG7A" = true ] && [ "$RUN_FIG8B" = true ]; then
        ESTIMATED_TIME="3-6 hours"
    elif [ "$RUN_FIG7A" = true ]; then
        ESTIMATED_TIME="2-4 hours"
    elif [ "$RUN_FIG8B" = true ]; then
        ESTIMATED_TIME="1-3 hours"
    fi
    
    log_info "Estimated total time: $ESTIMATED_TIME depending on hardware"
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
        if [ "${RUN_FIG7A_TRAINING:-true}" = true ]; then
            run_fig7a_training
        fi
        if [ "${RUN_FIG7A_INFERENCE:-true}" = true ]; then
            run_fig7a_inference
        fi
    else
        log_info "Skipping Figure 7a experiments"
    fi
    
    if [ "$RUN_FIG7B" = true ]; then
        if [ "${RUN_FIG7B_TRAINING:-true}" = true ]; then
            run_fig7b_training
        fi
        if [ "${RUN_FIG7B_INFERENCE:-true}" = true ]; then
            run_fig7b_inference
        fi
    else
        log_info "Skipping Figure 7b experiments"
    fi
    
    if [ "$RUN_FIG8B" = true ]; then
        if [ "${RUN_FIG8B_TRAINING:-true}" = true ]; then
            run_fig8b_training
        fi
        if [ "${RUN_FIG8B_INFERENCE:-true}" = true ]; then
            run_fig8b_inference
        fi
    else
        log_info "Skipping Figure 8b experiments"
    fi
    
    # Summary
    END_TIME=$(date)
    log_section "Experiment Suite Complete!"
    
    log_success "Selected experiments completed successfully!"
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
    
    # Show restart instructions
    echo
    log_info "🔄 To restart from a specific point:"
    log_info "  - Run only Fig 7a: $0 --fig7a"
    log_info "  - Run only Fig 7b: $0 --fig7b"  
    log_info "  - Run only Fig 8b: $0 --fig8b"
    log_info "  - From Fig 7b onwards: $0 --from-7b"
    log_info "  - Skip setup: $0 --fig7a --skip-setup"
    
    echo
    log_success "Thank you for reproducing our ViT results! 🎉"
}

# Handle interruption gracefully
trap 'log_warning "Script interrupted by user"; exit 130' INT

# Run main function
main "$@" 