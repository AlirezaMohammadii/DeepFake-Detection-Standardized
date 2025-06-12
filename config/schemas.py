"""
Comprehensive Configuration Schemas for Deepfake Detection Framework
Uses Pydantic for validation and type checking
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import torch
import psutil
import warnings

class DeviceType(str, Enum):
    """Supported device types"""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon

class InferenceMethod(str, Enum):
    """Bayesian inference methods"""
    VARIATIONAL = "variational"
    MCMC = "mcmc"
    EXACT = "exact"

class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class OutputFormat(str, Enum):
    """Output formats"""
    CSV = "csv"
    JSON = "json"
    PICKLE = "pickle"

# 1. Device and Hardware Configuration
class DeviceConfig(BaseModel):
    """Device and hardware configuration"""
    
    # Device selection
    preferred_device: DeviceType = Field(DeviceType.AUTO, description="Preferred device for computation")
    device_fallback_order: List[DeviceType] = Field(
        [DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU], 
        description="Device fallback order if preferred device unavailable"
    )
    
    # GPU configuration
    gpu_memory_fraction: float = Field(0.8, ge=0.1, le=1.0, description="Fraction of GPU memory to use")
    gpu_device_ids: Optional[List[int]] = Field(None, description="Specific GPU device IDs to use")
    enable_mixed_precision: bool = Field(True, description="Enable mixed precision training/inference")
    
    # CPU configuration
    cpu_threads: Optional[int] = Field(None, ge=1, le=128, description="Number of CPU threads (None for auto)")
    cpu_usage_threshold: float = Field(80.0, ge=10.0, le=100.0, description="CPU usage warning threshold (%)")
    cpu_critical_threshold: float = Field(90.0, ge=10.0, le=100.0, description="CPU usage critical threshold (%)")
    
    # Concurrency limits
    max_concurrent_processes: int = Field(4, ge=1, le=32, description="Maximum concurrent processes")
    cpu_concurrency_limit: int = Field(2, ge=1, le=8, description="Concurrency limit for CPU processing")
    gpu_concurrency_limit: int = Field(8, ge=1, le=32, description="Concurrency limit for GPU processing")
    
    @field_validator('cpu_critical_threshold')
    @classmethod
    def validate_cpu_thresholds(cls, v, info):
        if info.data and 'cpu_usage_threshold' in info.data and v <= info.data['cpu_usage_threshold']:
            raise ValueError("cpu_critical_threshold must be greater than cpu_usage_threshold")
        return v

# 2. Audio Processing Parameters
class AudioConfig(BaseModel):
    """Audio processing configuration"""
    
    # Basic audio parameters
    sample_rate: int = Field(16000, ge=8000, le=48000, description="Target sample rate for audio processing")
    chunk_size: int = Field(4096, ge=512, le=16384, description="Audio chunk size for processing")
    segment_duration_s: float = Field(2.0, ge=0.5, le=10.0, description="Default segment duration in seconds")
    
    # Spectral analysis parameters
    n_mels: int = Field(80, ge=40, le=128, description="Number of Mel bands for Mel spectrogram")
    n_fft: int = Field(512, ge=256, le=2048, description="FFT size")
    hop_length: int = Field(160, ge=80, le=512, description="Hop length for STFT")
    win_length: int = Field(400, ge=200, le=1024, description="Window length for STFT")
    
    # LFCC parameters
    n_lfcc: int = Field(20, ge=12, le=40, description="Number of LFCCs")
    
    # Processing parameters
    overlap_ratio: float = Field(0.75, ge=0.0, le=0.9, description="Overlap ratio for windowed analysis")
    preemphasis_coeff: float = Field(0.97, ge=0.9, le=0.99, description="Pre-emphasis coefficient")
    
    # Audio validation
    min_duration_s: float = Field(0.1, ge=0.01, le=1.0, description="Minimum audio duration")
    max_duration_s: float = Field(30.0, ge=1.0, le=300.0, description="Maximum audio duration")
    silence_threshold: float = Field(0.01, ge=0.001, le=0.1, description="Silence detection threshold")
    
    @field_validator('hop_length')
    @classmethod
    def validate_hop_length(cls, v, info):
        if info.data and 'sample_rate' in info.data:
            max_hop = info.data['sample_rate'] // 50
            if v > max_hop:
                warnings.warn(f"hop_length {v} may be too large for sample_rate {info.data['sample_rate']}")
        return v
    
    @field_validator('n_fft')
    @classmethod
    def validate_n_fft(cls, v, info):
        if info.data and 'win_length' in info.data and v < info.data['win_length']:
            next_pow2 = 1
            while next_pow2 < info.data['win_length']:
                next_pow2 *= 2
            warnings.warn(f"n_fft {v} adjusted to {next_pow2} to be >= win_length {info.data['win_length']}")
            return next_pow2
        return v

# 3. Security and Resource Limits
class SecurityConfig(BaseModel):
    """Security and resource limit configuration"""
    
    # Memory limits
    max_memory_gb: float = Field(8.0, ge=0.5, le=128.0, description="Maximum memory usage in GB")
    memory_warning_threshold_gb: float = Field(6.0, ge=0.5, le=128.0, description="Memory warning threshold in GB")
    enable_memory_monitoring: bool = Field(True, description="Enable memory usage monitoring")
    
    # Processing timeouts
    max_processing_time_s: float = Field(300.0, ge=10.0, le=3600.0, description="Maximum processing time per file")
    model_load_timeout_s: float = Field(60.0, ge=10.0, le=300.0, description="Model loading timeout")
    inference_timeout_s: float = Field(30.0, ge=1.0, le=300.0, description="Inference timeout per batch")
    
    # File constraints
    max_file_size_mb: float = Field(100.0, ge=1.0, le=1000.0, description="Maximum file size in MB")
    max_filename_length: int = Field(255, ge=50, le=500, description="Maximum filename length")
    max_path_length: int = Field(260, ge=100, le=1000, description="Maximum path length")
    
    # Security policies
    allowed_file_extensions: List[str] = Field(
        [".wav", ".mp3", ".flac", ".m4a", ".ogg"], 
        description="Allowed audio file extensions"
    )
    enable_file_validation: bool = Field(True, description="Enable file header validation")
    enable_path_traversal_protection: bool = Field(True, description="Prevent path traversal attacks")
    quarantine_suspicious_files: bool = Field(True, description="Quarantine suspicious files")
    
    # Resource monitoring
    monitoring_interval_s: float = Field(1.0, ge=0.1, le=10.0, description="Resource monitoring interval")
    enable_resource_limiting: bool = Field(True, description="Enable resource limiting")
    
    @field_validator('memory_warning_threshold_gb')
    @classmethod
    def validate_memory_thresholds(cls, v, info):
        if info.data and 'max_memory_gb' in info.data and v >= info.data['max_memory_gb']:
            raise ValueError("memory_warning_threshold_gb must be less than max_memory_gb")
        return v

# 4. Bayesian Network Parameters
class BayesianConfig(BaseModel):
    """Bayesian network configuration"""
    
    # Core engine settings
    enable_temporal_modeling: bool = Field(True, description="Enable temporal Bayesian modeling")
    enable_hierarchical_modeling: bool = Field(True, description="Enable hierarchical Bayesian modeling")
    enable_causal_analysis: bool = Field(True, description="Enable causal analysis")
    inference_method: InferenceMethod = Field(InferenceMethod.VARIATIONAL, description="Inference method")
    
    # Temporal modeling
    temporal_window_size: int = Field(10, ge=5, le=50, description="Temporal analysis window size")
    max_temporal_history: int = Field(20, ge=10, le=100, description="Maximum temporal history to keep")
    
    # Probability thresholds
    prior_authentic_prob: float = Field(0.7, ge=0.1, le=0.9, description="Prior probability of authenticity")
    uncertainty_threshold: float = Field(0.1, ge=0.01, le=0.5, description="Uncertainty threshold for decisions")
    confidence_threshold: float = Field(0.8, ge=0.5, le=0.99, description="Confidence threshold for high-confidence decisions")
    
    # Physics feature thresholds
    delta_fr_low_threshold: float = Field(0.3, ge=0.1, le=1.0, description="Delta FR low threshold")
    delta_fr_high_threshold: float = Field(0.7, ge=0.1, le=1.0, description="Delta FR high threshold")
    delta_ft_low_threshold: float = Field(0.01, ge=0.001, le=0.1, description="Delta FT low threshold")
    delta_ft_high_threshold: float = Field(0.05, ge=0.001, le=0.1, description="Delta FT high threshold")
    delta_fv_low_threshold: float = Field(0.2, ge=0.01, le=1.0, description="Delta FV low threshold")
    delta_fv_high_threshold: float = Field(0.5, ge=0.01, le=1.0, description="Delta FV high threshold")
    
    # Hierarchical modeling weights
    user_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for user-level evidence")
    session_weight: float = Field(0.4, ge=0.0, le=1.0, description="Weight for session-level evidence")
    audio_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for audio-level evidence")
    
    # User profile parameters
    user_adaptation_rate: float = Field(0.1, ge=0.01, le=0.5, description="User profile adaptation rate")
    min_samples_for_adaptation: int = Field(5, ge=1, le=20, description="Minimum samples for user adaptation")
    session_timeout_s: int = Field(3600, ge=300, le=86400, description="User session timeout in seconds")
    
    # Risk assessment thresholds
    low_risk_threshold: float = Field(0.8, ge=0.5, le=0.95, description="Low risk classification threshold")
    medium_risk_threshold: float = Field(0.4, ge=0.1, le=0.8, description="Medium risk classification threshold")
    
    # Transition probabilities
    same_state_prob: float = Field(0.7, ge=0.1, le=0.9, description="Probability of staying in same state")
    adjacent_state_prob: float = Field(0.25, ge=0.05, le=0.5, description="Probability of moving to adjacent state")
    distant_state_prob: float = Field(0.05, ge=0.01, le=0.3, description="Probability of jumping to distant state")
    
    @model_validator(mode='after')
    def validate_weights_sum_to_one(self):
        weights = [self.user_weight, self.session_weight, self.audio_weight]
        if abs(sum(weights) - 1.0) > 0.01:
            raise ValueError("user_weight + session_weight + audio_weight must sum to 1.0")
        return self
    
    @model_validator(mode='after')
    def validate_transition_probs(self):
        probs = [self.same_state_prob, self.adjacent_state_prob, self.distant_state_prob]
        if abs(sum(probs) - 1.0) > 0.01:
            raise ValueError("Transition probabilities must sum to 1.0")
        return self

# 5. Feature Extraction Configuration
class FeatureExtractionConfig(BaseModel):
    """Feature extraction configuration"""
    
    # Batch processing
    batch_size: int = Field(8, ge=1, le=64, description="Batch size for feature extraction")
    enable_length_bucketing: bool = Field(True, description="Enable length-based batching")
    max_batch_duration_s: float = Field(60.0, ge=10.0, le=300.0, description="Maximum total duration per batch")
    
    # Feature types
    enable_hubert_features: bool = Field(True, description="Enable HuBERT embeddings")
    enable_mel_spectrogram: bool = Field(True, description="Enable Mel spectrogram features")
    enable_lfcc_features: bool = Field(True, description="Enable LFCC features")
    enable_physics_features: bool = Field(True, description="Enable physics-based features")
    
    # Physics feature parameters
    embedding_dim_for_physics: int = Field(1024, ge=256, le=4096, description="Embedding dimension for physics analysis")
    physics_window_size_ms: int = Field(100, ge=20, le=500, description="Physics analysis window size in ms")
    physics_overlap_ratio: float = Field(0.5, ge=0.0, le=0.9, description="Physics analysis overlap ratio")
    
    # Vibration analysis
    max_rotation_angle_rad: float = Field(0.1, ge=0.01, le=1.0, description="Maximum rotation angle for analysis")
    vibration_amplitude_scale: float = Field(0.05, ge=0.001, le=0.5, description="Vibration amplitude scaling factor")
    enable_bessel_features: bool = Field(True, description="Enable Bessel function features")
    max_pca_components: int = Field(8, ge=1, le=20, description="Maximum PCA components for rotational analysis")
    
    # Feature validation
    enable_feature_validation: bool = Field(True, description="Enable feature validation")
    nan_handling_strategy: str = Field("replace", description="Strategy for handling NaN values: 'replace', 'skip', 'error'")
    feature_normalization: bool = Field(True, description="Normalize extracted features")

# 6. Path and Directory Management
class PathConfig(BaseModel):
    """Path and directory configuration"""
    
    # Base directories
    data_dir: str = Field("data", description="Data directory path")
    results_dir: str = Field("results", description="Results directory path")
    cache_dir: str = Field("cache", description="Cache directory path")
    logs_dir: str = Field("logs", description="Logs directory path")
    models_dir: str = Field("models", description="Models directory path")
    quarantine_dir: str = Field("quarantine", description="Quarantine directory path")
    
    # Output file patterns
    results_filename_pattern: str = Field("physics_features_summary_{timestamp}.csv", description="Results filename pattern")
    log_filename_pattern: str = Field("deepfake_detection_{date}.log", description="Log filename pattern")
    cache_filename_pattern: str = Field("features_{hash}.pkl", description="Cache filename pattern")
    
    # Directory creation
    create_dirs_if_missing: bool = Field(True, description="Create directories if they don't exist")
    dir_permissions: int = Field(0o755, description="Directory permissions (octal)")
    file_permissions: int = Field(0o644, description="File permissions (octal)")
    
    @field_validator('dir_permissions', 'file_permissions', mode='before')
    @classmethod
    def parse_octal_permissions(cls, v):
        """Parse octal permissions from various formats"""
        if isinstance(v, str):
            # Handle string octal format like '0o755' or '755'
            if v.startswith('0o'):
                return int(v, 8)
            elif v.isdigit():
                return int(v, 8)
            else:
                raise ValueError(f"Invalid octal permission format: {v}")
        elif isinstance(v, int):
            return v
        else:
            raise ValueError(f"Permission must be string or int, got {type(v)}")

# 7. Model and Cache Configuration
class ModelConfig(BaseModel):
    """Model and cache configuration"""
    
    # Model paths
    hubert_model_path: str = Field("facebook/hubert-large-ls960-ft", description="HuBERT model path or HF identifier")
    hubert_processor_path: Optional[str] = Field(None, description="HuBERT processor path (if different from model)")
    local_models_dir: str = Field("models", description="Local models directory")
    
    # Cache settings
    enable_model_cache: bool = Field(True, description="Enable model caching")
    enable_feature_cache: bool = Field(True, description="Enable feature caching")
    cache_expiry_hours: int = Field(24, ge=1, le=168, description="Cache expiry time in hours")
    max_cache_size_gb: float = Field(5.0, ge=0.1, le=50.0, description="Maximum cache size in GB")
    cache_cleanup_threshold: float = Field(0.8, ge=0.5, le=0.95, description="Cache cleanup threshold (fraction of max size)")
    
    # Model loading
    model_load_timeout_s: float = Field(300.0, ge=30.0, le=600.0, description="Model loading timeout")
    enable_model_warm_up: bool = Field(True, description="Warm up models on startup")
    model_precision: str = Field("float32", description="Model precision: float32, float16, bfloat16")

# 8. Monitoring and Logging Configuration  
class MonitoringConfig(BaseModel):
    """Monitoring and logging configuration"""
    
    # Logging settings
    log_level: LogLevel = Field(LogLevel.INFO, description="Logging level")
    enable_console_logging: bool = Field(True, description="Enable console logging")
    enable_file_logging: bool = Field(True, description="Enable file logging")
    log_rotation_size_mb: int = Field(100, ge=1, le=1000, description="Log file rotation size in MB")
    log_backup_count: int = Field(5, ge=1, le=20, description="Number of log backup files to keep")
    
    # Monitoring intervals
    system_monitor_interval_s: float = Field(5.0, ge=0.5, le=60.0, description="System monitoring interval")
    health_check_interval_s: float = Field(30.0, ge=5.0, le=300.0, description="Health check interval")
    metrics_export_interval_s: float = Field(60.0, ge=10.0, le=600.0, description="Metrics export interval")
    
    # Health check thresholds
    memory_health_threshold: float = Field(0.85, ge=0.5, le=0.95, description="Memory usage health threshold")
    cpu_health_threshold: float = Field(0.85, ge=0.5, le=0.95, description="CPU usage health threshold")
    disk_health_threshold: float = Field(0.90, ge=0.5, le=0.98, description="Disk usage health threshold")
    
    # Alerting
    enable_alerts: bool = Field(False, description="Enable alerting system")
    alert_cooldown_s: int = Field(300, ge=60, le=3600, description="Alert cooldown period in seconds")

# 9. Physics Feature Configuration
class PhysicsConfig(BaseModel):
    """Physics feature calculation configuration"""
    
    # Dynamics thresholds (for discretization)
    dynamics_thresholds: Dict[str, List[float]] = Field(
        {
            "delta_fr_revised": [0.3, 0.7],  # [low_threshold, high_threshold]
            "delta_ft_revised": [0.01, 0.05],
            "delta_fv_revised": [0.2, 0.5]
        },
        description="Thresholds for discretizing physics features"
    )
    
    # Spectral analysis
    spectral_rolloff_threshold: float = Field(0.95, ge=0.8, le=0.99, description="Spectral rolloff threshold")
    phase_space_bins: int = Field(50, ge=20, le=200, description="Phase space discretization bins")
    phase_space_range: List[float] = Field([-2.0, 2.0], description="Phase space analysis range")
    
    # Bessel analysis
    enable_bessel_analysis: bool = Field(True, description="Enable Bessel function analysis")
    bessel_order_max: int = Field(10, ge=3, le=20, description="Maximum Bessel function order")
    
    # Velocity analysis
    velocity_smoothing_window: int = Field(5, ge=1, le=20, description="Velocity smoothing window size")
    doppler_frequency_range: List[float] = Field([50.0, 8000.0], description="Doppler frequency analysis range")

# Main Configuration Class
class SystemConfig(BaseModel):
    """Main system configuration combining all subsystems"""
    
    # Configuration metadata
    config_version: str = Field("1.0.0", description="Configuration schema version")
    environment: str = Field("development", description="Environment: development, staging, production")
    
    # Subsystem configurations
    device: DeviceConfig = DeviceConfig()
    audio: AudioConfig = AudioConfig()
    security: SecurityConfig = SecurityConfig()
    bayesian: BayesianConfig = BayesianConfig()
    features: FeatureExtractionConfig = FeatureExtractionConfig()
    paths: PathConfig = PathConfig()
    models: ModelConfig = ModelConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    physics: PhysicsConfig = PhysicsConfig()
    
    # Output configuration
    output_format: OutputFormat = Field(OutputFormat.CSV, description="Default output format")
    enable_visualization: bool = Field(True, description="Enable result visualization")
    save_intermediate_results: bool = Field(False, description="Save intermediate processing results")
    
    class Config:
        env_nested_delimiter = '__'
        case_sensitive = False
        extra = 'forbid'  # Prevent unknown configuration keys
        
    @model_validator(mode='after')
    def validate_system_compatibility(self):
        """Validate cross-system compatibility"""
        # Estimate memory usage per batch
        estimated_memory_per_sample = 0.1  # GB per sample (rough estimate)
        estimated_batch_memory = self.features.batch_size * estimated_memory_per_sample
        
        if estimated_batch_memory > self.security.max_memory_gb * 0.8:
            warnings.warn(
                f"Batch size ({self.features.batch_size}) may exceed memory limits "
                f"({self.security.max_memory_gb} GB). Consider reducing batch size."
            )
        
        return self 