#!/usr/bin/env python3
"""
Configuration Manager for Radio-Mapper Emergency Response System

Handles loading, validation, and access to configuration settings
from config.yaml file with fallback defaults.
"""

import yaml
import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import socket

logger = logging.getLogger(__name__)

@dataclass
class BuoyConfig:
    """Buoy identification and location configuration"""
    id: str
    name: str
    latitude: float
    longitude: float
    altitude: float = 0.0

@dataclass
class GPSConfig:
    """GPS device and timing configuration"""
    enabled: bool
    device: str
    backup_device: str
    timeout_seconds: int
    use_fallback_location: bool

@dataclass
class SDRConfig:
    """SDR device configuration"""
    device_index: int
    sample_rate: int
    center_frequency_mhz: float
    gain: str
    ppm_error: int

@dataclass
class ServerConfig:
    """Central server configuration"""
    websocket_url: str
    http_url: str
    bind_host: str
    websocket_port: int
    http_port: int

@dataclass
class TimingConfig:
    """Time synchronization configuration"""
    method: str
    gps_enabled: bool
    ntp_enabled: bool
    ntp_servers: List[str]
    target_accuracy_us: int
    max_acceptable_us: int
    pps_device: str

class ConfigManager:
    """Manages configuration loading and access"""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.config = self._load_config()
        self._validate_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with defaults"""
        default_config = self._get_default_config()
        
        if not os.path.exists(self.config_file):
            logger.warning(f"Config file {self.config_file} not found, using defaults")
            return default_config
            
        try:
            with open(self.config_file, 'r') as f:
                user_config = yaml.safe_load(f)
                
            # Merge user config with defaults
            merged_config = self._deep_merge(default_config, user_config)
            logger.info(f"Loaded configuration from {self.config_file}")
            return merged_config
            
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            logger.warning("Using default configuration")
            return default_config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return {
            'buoy': {
                'id': 'OKC_BUOY_1',
                'name': 'Oklahoma City Buoy',
                'location': {
                    'latitude': 35.4676,
                    'longitude': -97.5164,
                    'altitude': 365.76
                },
                'gps': {
                    'enabled': True,
                    'device': '/dev/ttyUSB0',
                    'backup_device': '/dev/ttyACM0',
                    'timeout_seconds': 30,
                    'use_fallback_location': True
                }
            },
            'sdr': {
                'device_index': 0,
                'sample_rate': 2048000,
                'center_frequency_mhz': 121.5,
                'gain': 'auto',
                'ppm_error': 0
            },
            'central_server': {
                'websocket_url': 'ws://localhost:8081',
                'http_url': 'http://localhost:5001',
                'bind_host': '0.0.0.0',
                'websocket_port': 8081,
                'http_port': 5001
            },
            'timing': {
                'method': 'gps',
                'gps_timing': {
                    'enabled': True,
                    'pulse_per_second': True,
                    'pps_device': '/dev/pps0'
                },
                'ntp': {
                    'enabled': True,
                    'servers': ['time.nist.gov', 'pool.ntp.org'],
                    'sync_interval_minutes': 10
                },
                'target_accuracy_microseconds': 1,
                'max_acceptable_microseconds': 100
            },
            'signal_detection': {
                'power_threshold_dbm': -70,
                'confidence_threshold': 0.6,
                'emergency_frequencies': [121.5, 243.0, 406.025, 156.8],
                'fft_size': 1024,
                'overlap': 0.5,
                'correlation_window_seconds': 5.0
            },
            'tdoa': {
                'minimum_buoys': 3,
                'maximum_baseline_km': 50,
                'minimum_snr_db': 10,
                'confidence_threshold': 0.7
            },
            'logging': {
                'level': 'INFO',
                'file': 'radio-mapper.log'
            },
            'development': {
                'simulate_gps': False,
                'mock_sdr': False,
                'debug_timing': False
            }
        }
    
    def _deep_merge(self, default: Dict, user: Dict) -> Dict:
        """Deep merge user config into default config"""
        result = default.copy()
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _validate_config(self):
        """Validate configuration values"""
        try:
            # Validate buoy configuration
            buoy = self.config['buoy']
            assert isinstance(buoy['id'], str) and len(buoy['id']) > 0
            assert -90 <= buoy['location']['latitude'] <= 90
            assert -180 <= buoy['location']['longitude'] <= 180
            
            # Validate SDR configuration
            sdr = self.config['sdr']
            assert sdr['sample_rate'] > 0
            assert sdr['center_frequency_mhz'] > 0
            assert sdr['device_index'] >= 0
            
            # Validate server configuration
            server = self.config['central_server']
            assert 1024 <= server['websocket_port'] <= 65535
            assert 1024 <= server['http_port'] <= 65535
            assert server['websocket_port'] != server['http_port']
            
            # Validate timing configuration
            timing = self.config['timing']
            assert timing['method'] in ['gps', 'ntp', 'ptp', 'system']
            assert timing['target_accuracy_microseconds'] > 0
            
            logger.info("Configuration validation passed")
            
        except (KeyError, AssertionError, TypeError) as e:
            logger.error(f"Configuration validation failed: {e}")
            raise ValueError(f"Invalid configuration: {e}")
    
    def get_buoy_config(self) -> BuoyConfig:
        """Get buoy configuration"""
        buoy = self.config['buoy']
        location = buoy['location']
        
        return BuoyConfig(
            id=buoy['id'],
            name=buoy['name'],
            latitude=location['latitude'],
            longitude=location['longitude'],
            altitude=location.get('altitude', 0.0)
        )
    
    def get_gps_config(self) -> GPSConfig:
        """Get GPS configuration"""
        gps = self.config['buoy']['gps']
        
        return GPSConfig(
            enabled=gps['enabled'],
            device=gps['device'],
            backup_device=gps['backup_device'],
            timeout_seconds=gps['timeout_seconds'],
            use_fallback_location=gps['use_fallback_location']
        )
    
    def get_sdr_config(self) -> SDRConfig:
        """Get SDR configuration"""
        sdr = self.config['sdr']
        
        return SDRConfig(
            device_index=sdr['device_index'],
            sample_rate=sdr['sample_rate'],
            center_frequency_mhz=sdr['center_frequency_mhz'],
            gain=sdr['gain'],
            ppm_error=sdr['ppm_error']
        )
    
    def get_server_config(self) -> ServerConfig:
        """Get server configuration"""
        server = self.config['central_server']
        
        return ServerConfig(
            websocket_url=server['websocket_url'],
            http_url=server['http_url'],
            bind_host=server['bind_host'],
            websocket_port=server['websocket_port'],
            http_port=server['http_port']
        )
    
    def get_timing_config(self) -> TimingConfig:
        """Get timing configuration"""
        timing = self.config['timing']
        gps_timing = timing.get('gps_timing', {})
        ntp = timing.get('ntp', {})
        
        return TimingConfig(
            method=timing['method'],
            gps_enabled=gps_timing.get('enabled', True),
            ntp_enabled=ntp.get('enabled', True),
            ntp_servers=ntp.get('servers', ['time.nist.gov']),
            target_accuracy_us=timing['target_accuracy_microseconds'],
            max_acceptable_us=timing['max_acceptable_microseconds'],
            pps_device=gps_timing.get('pps_device', '/dev/pps0')
        )
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated path"""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def is_development_mode(self) -> bool:
        """Check if running in development mode"""
        return self.get('development.mock_sdr', False) or \
               self.get('development.simulate_gps', False)
    
    def get_emergency_frequencies(self) -> List[float]:
        """Get list of emergency frequencies to monitor"""
        return self.get('signal_detection.emergency_frequencies', [121.5, 243.0])
    
    def setup_logging(self):
        """Setup logging based on configuration"""
        log_level = self.get('logging.level', 'INFO')
        log_file = self.get('logging.file', 'radio-mapper.log')
        
        # Configure root logger
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
        
        # Set root logger level
        logging.getLogger().setLevel(numeric_level)
        
        logger.info(f"Logging configured: level={log_level}, file={log_file}")
    
    def auto_detect_interfaces(self) -> Dict[str, str]:
        """Auto-detect network interfaces and GPS devices"""
        result = {
            'ip_address': self._get_local_ip(),
            'gps_devices': self._detect_gps_devices(),
            'sdr_devices': self._detect_sdr_devices()
        }
        
        logger.info(f"Auto-detected interfaces: {result}")
        return result
    
    def _get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            # Connect to remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except:
            return "127.0.0.1"
    
    def _detect_gps_devices(self) -> List[str]:
        """Detect available GPS devices"""
        possible_devices = [
            '/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2',
            '/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyACM2',
            '/dev/ttyS0', '/dev/ttyS1'
        ]
        
        available = []
        for device in possible_devices:
            if os.path.exists(device):
                try:
                    # Basic check if device is accessible
                    with open(device, 'rb') as f:
                        available.append(device)
                except:
                    pass
        
        return available
    
    def _detect_sdr_devices(self) -> List[str]:
        """Detect available SDR devices"""
        try:
            import subprocess
            result = subprocess.run(['rtl_test', '-t'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Parse rtl_test output to count devices
                lines = result.stderr.split('\n')
                device_count = 0
                for line in lines:
                    if 'Found' in line and 'device' in line:
                        device_count += 1
                return [f"Device {i}" for i in range(device_count)]
        except:
            pass
        
        return []
    
    def generate_example_config(self, filename: str = "config.example.yaml"):
        """Generate an example configuration file"""
        try:
            with open(filename, 'w') as f:
                yaml.dump(self._get_default_config(), f, 
                         default_flow_style=False, sort_keys=False)
            logger.info(f"Generated example config: {filename}")
        except Exception as e:
            logger.error(f"Failed to generate example config: {e}")

# Global configuration instance
config = None

def get_config(config_file: str = "config.yaml") -> ConfigManager:
    """Get global configuration instance"""
    global config
    if config is None:
        config = ConfigManager(config_file)
    return config

def reload_config(config_file: str = "config.yaml"):
    """Reload configuration from file"""
    global config
    config = ConfigManager(config_file)
    return config 