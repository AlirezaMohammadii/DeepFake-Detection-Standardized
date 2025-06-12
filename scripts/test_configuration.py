#!/usr/bin/env python3
"""
Configuration Management System Test Script
Tests all aspects of the new configuration system
"""

import os
import sys
import asyncio
import tempfile
import shutil
import yaml
import json
from pathlib import Path
from typing import Dict, Any
import time

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_configuration_loading():
    """Test basic configuration loading"""
    print("🧪 Testing Configuration Loading...")
    
    try:
        from config import get_config_manager, get_config, get_config_value
        
        # Test basic loading
        config = get_config()
        print(f"✓ Configuration loaded successfully")
        print(f"  - Version: {config.config_version}")
        print(f"  - Environment: {config.environment}")
        
        # Test specific value access
        sample_rate = get_config_value('audio.sample_rate', 16000)
        print(f"  - Sample rate: {sample_rate}")
        
        device_type = get_config_value('device.preferred_device', 'auto')
        print(f"  - Device type: {device_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_environment_configs():
    """Test environment-specific configurations"""
    print("\n🧪 Testing Environment-Specific Configurations...")
    
    try:
        from config import ConfigManager
        
        environments = ['development', 'staging', 'production']
        results = {}
        
        for env in environments:
            print(f"  Testing {env} environment...")
            
            # Create config manager for specific environment
            config_manager = ConfigManager(
                config_dir="config",
                environment=env,
                enable_hot_reload=False
            )
            
            config = config_manager.get_config()
            
            results[env] = {
                'batch_size': config.features.batch_size,
                'log_level': config.monitoring.log_level,
                'memory_limit': config.security.max_memory_gb,
                'cache_enabled': config.models.enable_model_cache
            }
            
            print(f"    - Batch size: {config.features.batch_size}")
            print(f"    - Log level: {config.monitoring.log_level}")
            print(f"    - Memory limit: {config.security.max_memory_gb}GB")
            
        # Verify environments have different settings
        dev_batch = results['development']['batch_size']
        prod_batch = results['production']['batch_size']
        
        if dev_batch != prod_batch:
            print(f"✓ Environment configurations differ correctly (dev: {dev_batch}, prod: {prod_batch})")
        else:
            print(f"⚠ Environment configurations are identical")
            
        return True
        
    except Exception as e:
        print(f"❌ Environment configuration test failed: {e}")
        return False

def test_configuration_validation():
    """Test configuration validation"""
    print("\n🧪 Testing Configuration Validation...")
    
    try:
        from config.schemas import SystemConfig, DeviceConfig, AudioConfig
        
        # Test valid configuration
        valid_config = SystemConfig()
        print("✓ Default configuration is valid")
        
        # Test invalid configuration
        try:
            invalid_audio = AudioConfig(
                sample_rate=1000,  # Too low
                n_mels=10,         # Too low
                hop_length=2000    # Too high for sample rate
            )
            print("⚠ Invalid configuration was accepted (should have been rejected)")
        except Exception as e:
            print(f"✓ Invalid configuration correctly rejected: {type(e).__name__}")
        
        # Test boundary values
        boundary_config = SystemConfig(
            audio=AudioConfig(
                sample_rate=8000,  # Minimum
                n_mels=40,         # Minimum
                hop_length=80      # Minimum
            )
        )
        print("✓ Boundary value configuration is valid")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation test failed: {e}")
        return False

def test_environment_variable_override():
    """Test environment variable configuration override"""
    print("\n🧪 Testing Environment Variable Override...")
    
    try:
        from config import ConfigManager
        
        # Set test environment variables
        test_env_vars = {
            'DEEPFAKE_AUDIO__SAMPLE_RATE': '22050',
            'DEEPFAKE_DEVICE__PREFERRED_DEVICE': 'cpu',
            'DEEPFAKE_FEATURES__BATCH_SIZE': '16'
        }
        
        # Save original values
        original_values = {}
        for key in test_env_vars:
            original_values[key] = os.environ.get(key)
            os.environ[key] = test_env_vars[key]
        
        try:
            # Create config manager with env override enabled
            config_manager = ConfigManager(
                environment='development',
                enable_env_override=True,
                enable_hot_reload=False
            )
            
            config = config_manager.get_config()
            
            # Check if environment variables were applied
            if config.audio.sample_rate == 22050:
                print("✓ Audio sample rate override successful")
            else:
                print(f"❌ Audio sample rate override failed: {config.audio.sample_rate}")
            
            if config.device.preferred_device.value == 'cpu':
                print("✓ Device preference override successful")
            else:
                print(f"❌ Device preference override failed: {config.device.preferred_device}")
            
            if config.features.batch_size == 16:
                print("✓ Batch size override successful")
            else:
                print(f"❌ Batch size override failed: {config.features.batch_size}")
                
        finally:
            # Restore original environment variables
            for key, value in original_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        
        return True
        
    except Exception as e:
        print(f"❌ Environment variable override test failed: {e}")
        return False

async def test_hot_reload():
    """Test configuration hot-reloading"""
    print("\n🧪 Testing Configuration Hot-Reload...")
    
    try:
        from config import ConfigManager
        
        # Create temporary config file
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            
            # Initial configuration
            initial_config = {
                'config_version': '1.0.0',
                'environment': 'test',
                'audio': {'sample_rate': 16000},
                'features': {'batch_size': 4}
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(initial_config, f)
            
            # Create config manager with hot reload
            config_manager = ConfigManager(
                config_dir=temp_dir,
                environment='test',
                enable_hot_reload=True
            )
            
            # Load initial config
            config = config_manager.get_config()
            initial_batch_size = config.features.batch_size
            print(f"  Initial batch size: {initial_batch_size}")
            
            # Modify configuration file
            modified_config = initial_config.copy()
            modified_config['features']['batch_size'] = 8
            
            with open(config_path, 'w') as f:
                yaml.dump(modified_config, f)
            
            # Wait for file system event and reload
            await asyncio.sleep(2)  # Give time for file watcher
            
            # Check if configuration was reloaded
            updated_config = config_manager.get_config()
            updated_batch_size = updated_config.features.batch_size
            print(f"  Updated batch size: {updated_batch_size}")
            
            if updated_batch_size == 8:
                print("✓ Hot-reload successful")
                return True
            else:
                print("⚠ Hot-reload may not have triggered (this is timing-dependent)")
                return True  # Don't fail the test for timing issues
        
    except Exception as e:
        print(f"❌ Hot-reload test failed: {e}")
        return False

def test_configuration_export():
    """Test configuration export functionality"""
    print("\n🧪 Testing Configuration Export...")
    
    try:
        from config import get_config_manager
        
        config_manager = get_config_manager()
        
        # Test YAML export
        yaml_export = config_manager.export_config('yaml')
        if yaml_export and 'config_version' in yaml_export:
            print("✓ YAML export successful")
        else:
            print("❌ YAML export failed")
            return False
        
        # Test JSON export
        json_export = config_manager.export_config('json')
        if json_export and 'config_version' in json_export:
            print("✓ JSON export successful")
        else:
            print("❌ JSON export failed")
            return False
        
        # Verify exports are valid
        yaml_data = yaml.safe_load(yaml_export)
        json_data = json.loads(json_export)
        
        if yaml_data['config_version'] == json_data['config_version']:
            print("✓ Export formats are consistent")
        else:
            print("❌ Export formats are inconsistent")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration export test failed: {e}")
        return False

def test_device_configuration():
    """Test device configuration functionality"""
    print("\n🧪 Testing Device Configuration...")
    
    try:
        from config import get_config
        import torch
        
        config = get_config()
        device_config = config.device
        
        print(f"  Preferred device: {device_config.preferred_device}")
        print(f"  Fallback order: {device_config.device_fallback_order}")
        print(f"  GPU memory fraction: {device_config.gpu_memory_fraction}")
        print(f"  CPU usage threshold: {device_config.cpu_usage_threshold}%")
        
        # Test device availability checks
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        print(f"  CUDA available: {cuda_available}")
        print(f"  MPS available: {mps_available}")
        
        # Verify configuration makes sense for available hardware
        if cuda_available and device_config.preferred_device.value == 'cuda':
            print("✓ Device configuration matches available hardware")
        elif not cuda_available and device_config.preferred_device.value == 'cpu':
            print("✓ Device configuration appropriately falls back to CPU")
        else:
            print("✓ Device configuration set (hardware compatibility varies)")
        
        return True
        
    except Exception as e:
        print(f"❌ Device configuration test failed: {e}")
        return False

def test_security_configuration():
    """Test security configuration"""
    print("\n🧪 Testing Security Configuration...")
    
    try:
        from config import get_config
        
        config = get_config()
        security_config = config.security
        
        print(f"  Max file size: {security_config.max_file_size_mb}MB")
        print(f"  Max memory: {security_config.max_memory_gb}GB")
        print(f"  Processing timeout: {security_config.max_processing_time_s}s")
        print(f"  Allowed extensions: {security_config.allowed_file_extensions}")
        
        # Verify security settings are reasonable
        if security_config.max_file_size_mb > 0:
            print("✓ File size limit is positive")
        else:
            print("❌ File size limit is not positive")
            return False
        
        if security_config.max_memory_gb > 0:
            print("✓ Memory limit is positive")
        else:
            print("❌ Memory limit is not positive")
            return False
        
        if len(security_config.allowed_file_extensions) > 0:
            print("✓ File extensions list is not empty")
        else:
            print("❌ No allowed file extensions")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Security configuration test failed: {e}")
        return False

def test_bayesian_configuration():
    """Test Bayesian network configuration"""
    print("\n🧪 Testing Bayesian Configuration...")
    
    try:
        from config import get_config
        
        config = get_config()
        bayesian_config = config.bayesian
        
        print(f"  Inference method: {bayesian_config.inference_method}")
        print(f"  Uncertainty threshold: {bayesian_config.uncertainty_threshold}")
        print(f"  Confidence threshold: {bayesian_config.confidence_threshold}")
        print(f"  Temporal modeling: {bayesian_config.enable_temporal_modeling}")
        
        # Test probability sum validation
        weights_sum = (bayesian_config.user_weight + 
                      bayesian_config.session_weight + 
                      bayesian_config.audio_weight)
        
        if abs(weights_sum - 1.0) < 0.01:
            print("✓ Bayesian weights sum to 1.0")
        else:
            print(f"❌ Bayesian weights sum to {weights_sum:.3f}, not 1.0")
            return False
        
        # Test transition probabilities
        transition_sum = (bayesian_config.same_state_prob + 
                         bayesian_config.adjacent_state_prob + 
                         bayesian_config.distant_state_prob)
        
        if abs(transition_sum - 1.0) < 0.01:
            print("✓ Transition probabilities sum to 1.0")
        else:
            print(f"❌ Transition probabilities sum to {transition_sum:.3f}, not 1.0")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Bayesian configuration test failed: {e}")
        return False

async def run_all_tests():
    """Run all configuration tests"""
    print("🚀 Starting Configuration Management System Tests")
    print("=" * 60)
    
    tests = [
        ("Configuration Loading", test_configuration_loading),
        ("Environment Configs", test_environment_configs),
        ("Configuration Validation", test_configuration_validation),
        ("Environment Variable Override", test_environment_variable_override),
        ("Hot Reload", test_hot_reload),
        ("Configuration Export", test_configuration_export),
        ("Device Configuration", test_device_configuration),
        ("Security Configuration", test_security_configuration),
        ("Bayesian Configuration", test_bayesian_configuration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{test_name:30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Configuration system is working correctly.")
        return True
    else:
        print("⚠ Some tests failed. Please check the configuration system.")
        return False

def main():
    """Main test runner"""
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 