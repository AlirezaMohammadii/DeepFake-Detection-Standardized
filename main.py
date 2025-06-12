"""
Main entry point for the Deepfake Detection Framework
Adapted from the original test_runner.py to work with the new standardized structure
"""

import os
import sys
import asyncio
import pandas as pd
import torch
from tqdm import tqdm
import time
import traceback
from pathlib import Path
import pickle
from typing import List, Dict, Optional, Any, Tuple
import json
from datetime import datetime
import numpy as np
import re

# Setup path for imports - adapt to new structure
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    # First import the new configuration system
    from config import get_config_manager, get_config, get_config_value
    config_manager = get_config_manager()
    print("✓ New configuration system loaded")
    
    # Load configuration
    config = get_config()
    print(f"✓ Configuration loaded for environment: {config.environment}")
    
    # Adapted imports for new structure
    from data.loaders.audio_utils import load_audio
    from models.backbone.feature_extractor import ComprehensiveFeatureExtractor, RobustProcessor
    from inference.preprocessors.processing_pipeline import create_standard_pipeline, create_lightweight_pipeline
    from models.model_loader import DEVICE
    from utils.logging_system import create_project_logger
    from utils.security_validator import SecureAudioLoader, ResourceLimiter, InputValidator, SecurityConfig
    from data.processors.batch_processor import BatchProcessor, BatchConfig
    from utils.folder_manager import initialize_project_folders
    from models.ensemble.core.bayesian_engine import BayesianDeepfakeEngine, BayesianConfig, BayesianDetectionResult
    from models.ensemble.utils.temporal_buffer import TemporalFeatureBuffer
    from models.ensemble.utils.user_context import UserContextManager
    from models.ensemble.utils.uncertainty_estimation import PhysicsUncertaintyEstimator
    from models.ensemble.utils.causal_analysis import CausalFeatureAnalyzer
    print("✓ Enhanced security and batch processing modules loaded")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available")
    # Fall back to legacy imports if new config system isn't available
    try:
        from utils.config_loader import settings
        print("⚠ Falling back to legacy configuration system")
    except ImportError:
        print("⚠ No configuration system available")
    sys.exit(1)

# Configuration - load from config system
try:
    config = get_config()
    DATA_DIR = os.path.join(current_dir, config.paths.data_dir)
    RESULTS_DIR = os.path.join(current_dir, config.paths.results_dir)
    
    # Use configured filename pattern
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_pattern = config.paths.results_filename_pattern
    OUTPUT_CSV = os.path.join(RESULTS_DIR, filename_pattern.format(timestamp=timestamp))
    
    ERROR_LOG = os.path.join(RESULTS_DIR, "error_log.txt")
    
    print(f"✓ Configuration loaded:")
    print(f"  - Data directory: {DATA_DIR}")
    print(f"  - Results directory: {RESULTS_DIR}")
    print(f"  - Output file: {OUTPUT_CSV}")
    print(f"  - Environment: {config.environment}")
    
except Exception as e:
    print(f"⚠ Error loading configuration, using defaults: {e}")
    # Fallback to defaults
    DATA_DIR = os.path.join(current_dir, "data")
    RESULTS_DIR = os.path.join(current_dir, "results")
    OUTPUT_CSV = os.path.join(RESULTS_DIR, "physics_features_summary.csv")
    ERROR_LOG = os.path.join(RESULTS_DIR, "error_log.txt")

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# Import and run the rest of the functionality from the original test_runner
def main():
    """Main entry point for the deepfake detection framework"""
    print("Starting Deepfake Detection Framework...")
    print("Initializing with new standardized structure...")
    
    # Import the main execution from the scripts directory
    sys.path.insert(0, os.path.join(current_dir, 'scripts'))
    
    try:
        # Run the main functionality
        from test_runner import main as original_main
        asyncio.run(original_main())
    except Exception as e:
        print(f"Error running main functionality: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 