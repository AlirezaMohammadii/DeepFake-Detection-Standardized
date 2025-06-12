"""
Configuration Package for Deepfake Detection Framework
Provides centralized configuration management with hot-reloading and environment support
"""

from .config_manager import (
    ConfigManager,
    get_config_manager,
    get_config,
    get_config_value,
    reload_config,
    ConfigSource,
    ConfigMetadata
)

from .schemas import (
    SystemConfig,
    DeviceConfig,
    AudioConfig,
    SecurityConfig,
    BayesianConfig,
    FeatureExtractionConfig,
    PathConfig,
    ModelConfig,
    MonitoringConfig,
    PhysicsConfig,
    DeviceType,
    InferenceMethod,
    LogLevel,
    OutputFormat
)

__all__ = [
    # Configuration Manager
    'ConfigManager',
    'get_config_manager',
    'get_config',
    'get_config_value',
    'reload_config',
    'ConfigSource',
    'ConfigMetadata',
    
    # Configuration Schemas
    'SystemConfig',
    'DeviceConfig',
    'AudioConfig',
    'SecurityConfig',
    'BayesianConfig',
    'FeatureExtractionConfig',
    'PathConfig',
    'ModelConfig',
    'MonitoringConfig',
    'PhysicsConfig',
    
    # Enums
    'DeviceType',
    'InferenceMethod',
    'LogLevel',
    'OutputFormat'
]

# Version information
__version__ = "1.0.0"
