#!/bin/bash

# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Setup and test script for ViT experiments
# This script will:
# 1. Set up conda environment (if needed)
# 2. Verify setup  
# 3. Run basic tests
# 4. Give instructions for running full experiments

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
        log_warning "Quick test had issues, but setup is likely OK"
    fi
}

# Main execution
main() {
    log_section "ViT Experiments - Setup and Test"
    
    log_info "This script will set up your environment and run basic tests"
    log_info "Estimated time: 5-15 minutes"
    
    # Check if we're in the right directory
    if [ ! -f ".project-root" ]; then
        log_error ".project-root file not found"
        log_info "Please run this script from: tbp.tbs_sensorimotor_intelligence/pytorch"
        exit 1
    fi
    
    # Setup and verification
    check_conda
    setup_environment
    verify_setup
    run_quick_test
    
    # Success message and next steps
    log_section "Setup Complete!"
    
    log_success "Environment is ready for ViT experiments!"
    
    echo
    log_info "🎯 Next Steps - Choose your experiment suite:"
    echo
    log_info "🚀 Run ALL experiments:"
    echo "   ./scripts/run_all_experiments.sh      # Complete reproduction suite"
    echo
    log_info "📊 Monitor results:"
    echo "   - WandB dashboard: https://wandb.ai/your-team/benchmark_vit"
    echo "   - Local logs: ~/tbp/results/dmc/results/vit/logs_reproduction"
    echo
    log_success "Happy experimenting! 🎉"
}

# Handle interruption gracefully
trap 'log_warning "Script interrupted by user"; exit 130' INT

# Run main function
main "$@" 