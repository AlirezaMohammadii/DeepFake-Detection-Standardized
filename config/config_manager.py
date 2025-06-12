"""
Advanced Configuration Manager with Hot-Reloading and Environment Support
Supports YAML, JSON, and environment variable configuration sources
"""

import os
import yaml
import json
import time
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import hashlib

from .schemas import SystemConfig, DeviceType, LogLevel

logger = logging.getLogger(__name__)

class ConfigSource(str, Enum):
    """Configuration source types"""
    FILE = "file"
    ENVIRONMENT = "environment"
    REMOTE = "remote"
    RUNTIME = "runtime"

@dataclass
class ConfigMetadata:
    """Configuration metadata for tracking and validation"""
    source: ConfigSource
    last_updated: datetime
    version: str
    checksum: str
    environment: str
    validation_errors: List[str] = field(default_factory=list)
    
class ConfigWatcher(FileSystemEventHandler):
    """File system watcher for configuration changes"""
    
    def __init__(self, config_manager: 'ConfigManager', config_path: Path):
        self.config_manager = config_manager
        self.config_path = config_path
        self.last_modified = time.time()
        
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
            
        # Check if it's our config file
        if Path(event.src_path) == self.config_path:
            # Debounce rapid file changes
            current_time = time.time()
            if current_time - self.last_modified > 1.0:  # 1 second debounce
                self.last_modified = current_time
                logger.info(f"Configuration file changed: {event.src_path}")
                asyncio.create_task(self.config_manager.reload_config())

class ConfigManager:
    """
    Advanced configuration manager with hot-reloading, validation, and multi-source support
    """
    
    def __init__(self, 
                 config_dir: Union[str, Path] = "config",
                 environment: str = "development",
                 enable_hot_reload: bool = True,
                 enable_env_override: bool = True,
                 config_cache_ttl_seconds: int = 300):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Configuration directory path
            environment: Environment name (development, staging, production)
            enable_hot_reload: Enable hot-reloading of configuration files
            enable_env_override: Allow environment variables to override config
            config_cache_ttl_seconds: Configuration cache TTL in seconds
        """
        self.config_dir = Path(config_dir)
        self.environment = environment
        self.enable_hot_reload = enable_hot_reload
        self.enable_env_override = enable_env_override
        self.cache_ttl = timedelta(seconds=config_cache_ttl_seconds)
        
        # Configuration state
        self._config: Optional[SystemConfig] = None
        self._config_metadata: Optional[ConfigMetadata] = None
        self._config_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._change_callbacks: List[Callable[[SystemConfig], None]] = []
        
        # File watching
        self._observer: Optional[Observer] = None
        self._watcher: Optional[ConfigWatcher] = None
        
        # Thread safety
        self._config_lock = threading.RLock()
        
        # Initialize
        self._setup_config_directory()
        
    def _setup_config_directory(self):
        """Setup configuration directory structure"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create environment-specific subdirectories
        for env in ["development", "staging", "production"]:
            (self.config_dir / env).mkdir(exist_ok=True)
            
        # Create schemas directory for validation
        (self.config_dir / "schemas").mkdir(exist_ok=True)
        
    def get_config_path(self, filename: str = "config.yaml") -> Path:
        """Get configuration file path for current environment"""
        # Try environment-specific config first
        env_specific_path = self.config_dir / self.environment / filename
        if env_specific_path.exists():
            return env_specific_path
            
        # Fall back to general config
        general_path = self.config_dir / filename
        if general_path.exists():
            return general_path
            
        # Return environment-specific path for creation
        return env_specific_path
    
    def _calculate_checksum(self, content: str) -> str:
        """Calculate SHA-256 checksum of content"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _load_yaml_file(self, filepath: Path) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                data = yaml.safe_load(content)
                
            if data is None:
                data = {}
                
            return data
        except Exception as e:
            logger.error(f"Error loading YAML file {filepath}: {e}")
            raise
    
    def _load_json_file(self, filepath: Path) -> Dict[str, Any]:
        """Load JSON configuration file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file {filepath}: {e}")
            raise
    
    def _load_environment_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config_dict = {}
        
        # Look for environment variables with specific prefixes
        prefixes = ['DEEPFAKE_', 'DF_', 'CONFIG_']
        
        for key, value in os.environ.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    # Remove prefix and convert to nested dict
                    config_key = key[len(prefix):].lower()
                    
                    # Handle nested keys (e.g., DEEPFAKE_DEVICE__PREFERRED_DEVICE)
                    if '__' in config_key:
                        parts = config_key.split('__')
                        current = config_dict
                        
                        for part in parts[:-1]:
                            if part not in current:
                                current[part] = {}
                            current = current[part]
                        
                        # Convert value to appropriate type
                        current[parts[-1]] = self._convert_env_value(value)
                    else:
                        config_dict[config_key] = self._convert_env_value(value)
                    break
        
        return config_dict
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable value to appropriate Python type"""
        # Boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # None/null values
        if value.lower() in ('none', 'null', ''):
            return None
        
        # Numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # List values (comma-separated)
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        # String value
        return value
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    async def load_config(self, config_file: str = "config.yaml") -> SystemConfig:
        """
        Load configuration from file and environment variables
        
        Args:
            config_file: Configuration file name
            
        Returns:
            Validated SystemConfig instance
        """
        with self._config_lock:
            try:
                config_path = self.get_config_path(config_file)
                logger.info(f"Loading configuration from: {config_path}")
                
                # Load base configuration
                config_dict = {}
                
                if config_path.exists():
                    if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                        config_dict = self._load_yaml_file(config_path)
                    elif config_path.suffix.lower() == '.json':
                        config_dict = self._load_json_file(config_path)
                    else:
                        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
                else:
                    logger.warning(f"Configuration file not found: {config_path}. Using defaults.")
                
                # Merge with environment variables if enabled
                if self.enable_env_override:
                    env_config = self._load_environment_config()
                    if env_config:
                        logger.info("Merging environment variable overrides")
                        config_dict = self._merge_configs(config_dict, env_config)
                
                # Set environment if not specified
                if 'environment' not in config_dict:
                    config_dict['environment'] = self.environment
                
                # Validate and create configuration object
                try:
                    config = SystemConfig(**config_dict)
                    validation_errors = []
                except Exception as e:
                    logger.error(f"Configuration validation failed: {e}")
                    # Create default config and log errors
                    config = SystemConfig()
                    validation_errors = [str(e)]
                
                # Create metadata
                content_str = json.dumps(config_dict, sort_keys=True)
                metadata = ConfigMetadata(
                    source=ConfigSource.FILE,
                    last_updated=datetime.now(),
                    version=config.config_version,
                    checksum=self._calculate_checksum(content_str),
                    environment=self.environment,
                    validation_errors=validation_errors
                )
                
                # Update internal state
                self._config = config
                self._config_metadata = metadata
                
                # Setup file watching if enabled
                if self.enable_hot_reload and config_path.exists():
                    await self._setup_file_watching(config_path)
                
                # Notify callbacks
                await self._notify_config_change(config)
                
                logger.info(f"Configuration loaded successfully (version: {config.config_version})")
                return config
                
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                # Return default configuration
                if self._config is None:
                    self._config = SystemConfig()
                return self._config
    
    async def _setup_file_watching(self, config_path: Path):
        """Setup file system watching for configuration changes"""
        try:
            if self._observer:
                self._observer.stop()
                self._observer.join()
            
            self._watcher = ConfigWatcher(self, config_path)
            self._observer = Observer()
            self._observer.schedule(
                self._watcher, 
                str(config_path.parent), 
                recursive=False
            )
            self._observer.start()
            logger.info(f"Started configuration file watching: {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to setup file watching: {e}")
    
    async def reload_config(self):
        """Reload configuration from file"""
        logger.info("Reloading configuration...")
        try:
            await self.load_config()
            logger.info("Configuration reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
    
    def get_config(self) -> SystemConfig:
        """Get current configuration (synchronous)"""
        with self._config_lock:
            if self._config is None:
                # Load synchronously if not loaded
                import asyncio
                try:
                    # Try to get existing event loop
                    loop = asyncio.get_running_loop()
                    # If we're in an async context, we can't run another event loop
                    # So we'll load synchronously
                    self._load_config_sync()
                except RuntimeError:
                    # No event loop running, safe to create one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        self._config = loop.run_until_complete(self.load_config())
                    finally:
                        loop.close()
            return self._config
    
    def _load_config_sync(self):
        """Load configuration synchronously"""
        try:
            config_path = self.get_config_path()
            logger.info(f"Loading configuration synchronously from: {config_path}")
            
            # Load base configuration
            config_dict = {}
            
            if config_path.exists():
                if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                    config_dict = self._load_yaml_file(config_path)
                elif config_path.suffix.lower() == '.json':
                    config_dict = self._load_json_file(config_path)
                else:
                    raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
            else:
                logger.warning(f"Configuration file not found: {config_path}. Using defaults.")
            
            # Merge with environment variables if enabled
            if self.enable_env_override:
                env_config = self._load_environment_config()
                if env_config:
                    logger.info("Merging environment variable overrides")
                    config_dict = self._merge_configs(config_dict, env_config)
            
            # Set environment if not specified
            if 'environment' not in config_dict:
                config_dict['environment'] = self.environment
            
            # Validate and create configuration object
            try:
                config = SystemConfig(**config_dict)
                validation_errors = []
            except Exception as e:
                logger.error(f"Configuration validation failed: {e}")
                # Create default config and log errors
                config = SystemConfig()
                validation_errors = [str(e)]
            
            # Create metadata
            content_str = json.dumps(config_dict, sort_keys=True)
            metadata = ConfigMetadata(
                source=ConfigSource.FILE,
                last_updated=datetime.now(),
                version=config.config_version,
                checksum=self._calculate_checksum(content_str),
                environment=self.environment,
                validation_errors=validation_errors
            )
            
            # Update internal state
            self._config = config
            self._config_metadata = metadata
            
            logger.info(f"Configuration loaded successfully (version: {config.config_version})")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Return default configuration
            if self._config is None:
                self._config = SystemConfig()
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key path
        
        Args:
            key_path: Dot-separated key path (e.g., 'device.preferred_device')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        config = self.get_config()
        
        # Navigate through nested structure
        current = config.dict()
        for key in key_path.split('.'):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def set_config_value(self, key_path: str, value: Any):
        """
        Set configuration value by dot-separated key path (runtime override)
        
        Args:
            key_path: Dot-separated key path
            value: Value to set
        """
        with self._config_lock:
            if self._config is None:
                self._config = self.get_config()
            
            # Create a mutable copy of the config
            config_dict = self._config.model_dump(mode='json')
            
            # Navigate to the parent of the target key
            current = config_dict
            keys = key_path.split('.')
            
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set the value
            current[keys[-1]] = value
            
            # Recreate the config object
            try:
                self._config = SystemConfig(**config_dict)
                logger.info(f"Configuration value updated: {key_path} = {value}")
            except Exception as e:
                logger.error(f"Failed to update configuration value: {e}")
    
    def add_change_callback(self, callback: Callable[[SystemConfig], None]):
        """Add callback for configuration changes"""
        self._change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: Callable[[SystemConfig], None]):
        """Remove configuration change callback"""
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)
    
    async def _notify_config_change(self, config: SystemConfig):
        """Notify all callbacks of configuration change"""
        for callback in self._change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(config)
                else:
                    callback(config)
            except Exception as e:
                logger.error(f"Error in configuration change callback: {e}")
    
    def save_config(self, config_file: str = "config.yaml"):
        """Save current configuration to file"""
        try:
            config_path = self.get_config_path(config_file)
            config_dict = self._config.model_dump(mode='json') if self._config else {}
            
            # Create directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
            
            logger.info(f"Configuration saved to: {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def get_metadata(self) -> Optional[ConfigMetadata]:
        """Get configuration metadata"""
        return self._config_metadata
    
    def validate_config(self) -> List[str]:
        """Validate current configuration and return errors"""
        try:
            if self._config is None:
                return ["Configuration not loaded"]
            
            # Re-validate the current configuration
            SystemConfig(**self._config.model_dump())
            return []
            
        except Exception as e:
            return [str(e)]
    
    def export_config(self, format: str = "yaml") -> str:
        """Export current configuration as string"""
        if self._config is None:
            return ""
        
        config_dict = self._config.model_dump(mode='json')  # Use JSON mode to handle enums properly
        
        if format.lower() == "yaml":
            return yaml.dump(config_dict, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            return json.dumps(config_dict, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def create_environment_config(self, environment: str, base_config: Optional[SystemConfig] = None):
        """Create environment-specific configuration file"""
        try:
            env_dir = self.config_dir / environment
            env_dir.mkdir(parents=True, exist_ok=True)
            
            config_path = env_dir / "config.yaml"
            
            if base_config is None:
                base_config = self._config or SystemConfig()
            
            # Customize config for environment
            config_dict = base_config.model_dump(mode='json')
            config_dict['environment'] = environment
            
            # Environment-specific adjustments
            if environment == "production":
                config_dict['monitoring']['log_level'] = "info"
                config_dict['security']['max_processing_time_s'] = 600.0
                config_dict['device']['enable_mixed_precision'] = True
            elif environment == "development":
                config_dict['monitoring']['log_level'] = "debug"
                config_dict['monitoring']['enable_console_logging'] = True
                config_dict['models']['enable_model_cache'] = False
            elif environment == "staging":
                config_dict['monitoring']['log_level'] = "info"
                config_dict['security']['max_processing_time_s'] = 300.0
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Environment configuration created: {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to create environment configuration: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            
        self._change_callbacks.clear()
        logger.info("Configuration manager cleaned up")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None

def get_config_manager(
    config_dir: Union[str, Path] = "config",
    environment: Optional[str] = None,
    **kwargs
) -> ConfigManager:
    """Get or create global configuration manager instance"""
    global _config_manager
    
    if _config_manager is None:
        if environment is None:
            environment = os.getenv('ENVIRONMENT', 'development')
        
        _config_manager = ConfigManager(
            config_dir=config_dir,
            environment=environment,
            **kwargs
        )
    
    return _config_manager

def get_config() -> SystemConfig:
    """Get current system configuration"""
    return get_config_manager().get_config()

def get_config_value(key_path: str, default: Any = None) -> Any:
    """Get configuration value by key path"""
    return get_config_manager().get_config_value(key_path, default)

async def reload_config():
    """Reload configuration"""
    await get_config_manager().reload_config() 