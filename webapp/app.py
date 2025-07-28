
from flask import Flask, render_template, jsonify, request
import yaml
import os
from datetime import datetime, timedelta
import sys
import serial
import time
import re
import requests
import json

# Add the parent directory to the path so we can import the config manager
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config_manager import ConfigManager

app = Flask(__name__)

# Load configuration using the actual config manager
config_manager = ConfigManager()
config = config_manager.config

# Central Processor API configuration
CENTRAL_PROCESSOR_HOST = os.getenv('CENTRAL_HOST', 'central-processor')
CENTRAL_PROCESSOR_PORT = os.getenv('CENTRAL_PORT', '5001')
CENTRAL_PROCESSOR_URL = f"http://{CENTRAL_PROCESSOR_HOST}:{CENTRAL_PROCESSOR_PORT}"

def get_central_processor_data(endpoint, default=None):
    """Get data from central processor API with fallback"""
    try:
        url = f"{CENTRAL_PROCESSOR_URL}/api/{endpoint}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Failed to connect to central processor: {e}")
    return default or []

# Cache for hardware detection to prevent inconsistent results
_hardware_cache = {
    'last_check': None,
    'interfaces': None,
    'cache_duration': 10  # seconds
}

def get_cached_interfaces():
    """Get hardware interfaces with caching to prevent inconsistent detection"""
    now = time.time()
    
    if (_hardware_cache['last_check'] is None or 
        now - _hardware_cache['last_check'] > _hardware_cache['cache_duration']):
        
        # Cache expired, refresh
        _hardware_cache['interfaces'] = config_manager.auto_detect_interfaces()
        _hardware_cache['last_check'] = now
    
    return _hardware_cache['interfaces']

def get_gps_coordinates():
    """Get real GPS coordinates from connected GPS device (fast, non-blocking)"""
    gps_config = config_manager.get_gps_config()
    
    if not gps_config.enabled:
        # GPS disabled, use fallback
        return (
            config.get('buoy.location.latitude', 35.55132013715708),
            config.get('buoy.location.longitude', -97.53221383761282),
            "disabled"
        )
    
    # Try to get REAL GPS coordinates from device
    device_path = gps_config.device
    actual_gps_coords = None
    
    if os.path.exists(device_path):
        # Try multiple baud rates for U-Blox GPS
        baud_rates = [9600, 4800, 38400]
        
        for baud_rate in baud_rates:
            try:
                with serial.Serial(device_path, baud_rate, timeout=0.2) as ser:
                    # Read several lines to find valid coordinates
                    for _ in range(8):
                        try:
                            line = ser.readline().decode('ascii', errors='ignore').strip()
                            
                            # Only accept GPRMC with 'A' status (valid fix) or GPGGA with coordinates
                            if line.startswith('$GPRMC') and ',A,' in line:
                                coords = parse_nmea_coordinates(line)
                                if coords:
                                    actual_gps_coords = coords
                                    break
                            elif line.startswith('$GPGGA') and line.count(',') >= 6:
                                parts = line.split(',')
                                if len(parts) >= 6 and parts[2] and parts[4] and parts[6] != '0':
                                    coords = parse_nmea_coordinates(line)
                                    if coords:
                                        actual_gps_coords = coords
                                        break
                        except:
                            continue
                
                if actual_gps_coords:
                    break  # Found valid GPS coordinates
                    
            except Exception as e:
                continue
    
    # Return actual GPS coordinates if found
    if actual_gps_coords:
        return actual_gps_coords[0], actual_gps_coords[1], "gps_locked"
    
    # If we can't read GPS, use fallback if enabled
    if gps_config.use_fallback_location:
        return (
            config.get('buoy.location.latitude', 35.55132013715708),
            config.get('buoy.location.longitude', -97.53221383761282),
            "fallback"
        )
    
    return None, None, "no_fix"

def parse_nmea_coordinates(nmea_sentence):
    """Parse latitude and longitude from NMEA sentence - only return if valid fix"""
    try:
        parts = nmea_sentence.split(',')
        
        if nmea_sentence.startswith('$GPGGA'):
            # GPGGA format: $GPGGA,time,lat,N/S,lon,E/W,quality,satellites,hdop,altitude,M,geoid,M,dgps_time,dgps_id*checksum
            # Quality: 0=invalid, 1=GPS fix, 2=DGPS fix
            if (len(parts) >= 7 and parts[2] and parts[4] and 
                parts[6] and int(parts[6]) > 0):  # Quality > 0 means valid fix
                
                lat_str = parts[2]
                lat_dir = parts[3]
                lon_str = parts[4]
                lon_dir = parts[5]
                
                # Convert from DDMM.MMMM format to decimal degrees
                if len(lat_str) >= 4 and len(lon_str) >= 5:
                    lat = float(lat_str[:2]) + float(lat_str[2:]) / 60.0
                    if lat_dir == 'S':
                        lat = -lat
                        
                    lon = float(lon_str[:3]) + float(lon_str[3:]) / 60.0
                    if lon_dir == 'W':
                        lon = -lon
                        
                    # Sanity check coordinates are reasonable
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        return lat, lon
                
        elif nmea_sentence.startswith('$GPRMC'):
            # GPRMC format: $GPRMC,time,status,lat,N/S,lon,E/W,speed,course,date,mag_var,var_dir*checksum
            # Status: A = valid fix, V = invalid
            if (len(parts) >= 7 and parts[3] and parts[5] and 
                parts[2] == 'A'):  # A = active/valid fix
                
                lat_str = parts[3]
                lat_dir = parts[4]
                lon_str = parts[5]
                lon_dir = parts[6]
                
                # Convert from DDMM.MMMM format to decimal degrees
                if len(lat_str) >= 4 and len(lon_str) >= 5:
                    lat = float(lat_str[:2]) + float(lat_str[2:]) / 60.0
                    if lat_dir == 'S':
                        lat = -lat
                        
                    lon = float(lon_str[:3]) + float(lon_str[3:]) / 60.0
                    if lon_dir == 'W':
                        lon = -lon
                        
                    # Sanity check coordinates are reasonable
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        return lat, lon
    except:
        pass
    
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/devices')
def get_devices():
    # First try to get real buoy data from central processor
    try:
        buoy_data = get_central_processor_data('nodes')
        if buoy_data and isinstance(buoy_data, list):
            # Convert central processor node data to device format
            devices = []
            for node_info in buoy_data:
                # Parse timestamp for better formatting
                last_seen = node_info.get('lastSeen', datetime.now().isoformat())
                try:
                    # Convert to more readable format
                    parsed_time = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
                    formatted_time = parsed_time.strftime('%Y-%m-%d %H:%M:%S UTC')
                except:
                    formatted_time = last_seen
                
                devices.append({
                    'id': node_info.get('id'),
                    'name': node_info.get('name', node_info.get('id')),
                    'lat': node_info.get('lat', 35.5513177334763),
                    'lng': node_info.get('lng', -97.53220535352492),
                    'status': node_info.get('status', 'active'),
                    'lastSeen': last_seen,
                    'lastSeenFormatted': formatted_time,
                    'hardware': 'RTL-SDR v3',  # Add hardware info
                    'location_source': 'gps_locked',  # Add location source
                    'capabilities': ['fm_scanner', 'emergency_scanner'],
                    'signal_count': 0  # Will be updated by frontend
                })
            return jsonify(devices)
    except Exception as e:
        print(f"Failed to get buoy data from central processor: {e}")
    
    # Fallback to development mode or local detection
    dev_mode = config_manager.is_development_mode()
    
    if dev_mode:
        # Return mock devices for development
        devices = [
            {'id': 'OKC_BUOY_1', 'name': 'OKC North', 'lat': 35.5200, 'lng': -97.5164, 'status': 'active', 'lastSeen': datetime.now().isoformat()},
            {'id': 'OKC_BUOY_2', 'name': 'OKC East', 'lat': 35.4676, 'lng': -97.4200, 'status': 'active', 'lastSeen': datetime.now().isoformat()},
            {'id': 'OKC_BUOY_3', 'name': 'OKC South', 'lat': 35.4100, 'lng': -97.5164, 'status': 'active', 'lastSeen': datetime.now().isoformat()},
        ]
    else:
        # Try to detect real hardware (with caching)
        interfaces = get_cached_interfaces()
        devices = []
        
        # If we have real SDR devices, create device entries for them
        if interfaces.get('sdr_devices'):
            # Get real GPS coordinates
            lat, lng, gps_status = get_gps_coordinates()
            
            if lat is not None and lng is not None:
                location_status = gps_status
            else:
                # No GPS data available
                lat, lng = 0.0, 0.0
                location_status = "no_gps"
            
            # Create device representing this local SDR
            devices = [
                {
                    'id': 1,
                    'name': f'Local SDR {config.get("buoy.name", "Unknown")}',
                    'lat': lat,
                    'lng': lng,
                    'status': 'active',
                    'lastSeen': datetime.now().isoformat(),
                    'hardware': 'real_sdr',
                    'location_source': location_status
                }
            ]
    
    return jsonify(devices)

@app.route('/api/signals')
def get_signals():
    # First try to get real signal data from central processor
    try:
        # Get individual detections (more useful than triangulated signals)
        signal_data = get_central_processor_data('detections')
        
        if signal_data:
            # Convert central processor signal data to web format
            signals = []
            for signal in signal_data:
                signals.append({
                    'id': signal.get('id'),
                    'frequency': signal.get('frequency_mhz', 0),
                    'signal_strength': signal.get('signal_strength_dbm', -100),
                    'lat': signal.get('lat', 35.5513177334763),
                    'lng': signal.get('lng', -97.53220535352492),
                    'detected_by': [signal.get('node_id', 'unknown')],
                    'timestamp': signal.get('timestamp', datetime.now().isoformat()),
                    'signal_type': signal.get('signal_type', 'Unknown'),
                    'confidence': signal.get('confidence', 0),
                    'triangulated': signal.get('triangulated', False)
                })
            return jsonify(signals)
    except Exception as e:
        print(f"Failed to get signal data from central processor: {e}")
    
    # Fallback to development mode
    dev_mode = config_manager.is_development_mode()
    
    if dev_mode:
        # Return mock signals for development
        signals = [
            {
                'id': 1,
                'frequency': 105.7,  # NPR MHz
                'signal_strength': -45,  # dBm
                'lat': 35.4676,
                'lng': -97.5164,
                'detected_by': ['OKC_BUOY_1', 'OKC_BUOY_2', 'OKC_BUOY_3'],
                'timestamp': datetime.now().isoformat(),
                'signal_type': 'FM'
            },
            {
                'id': 2,
                'frequency': 121.5,  # Emergency MHz
                'signal_strength': -72,  # dBm
                'lat': 35.5200,
                'lng': -97.4200,
                'detected_by': ['OKC_BUOY_1', 'OKC_BUOY_3'],
                'timestamp': datetime.now().isoformat(),
                'signal_type': 'Emergency'
            }
        ]
    else:
        # In production, return empty list if no central processor data
        signals = []
    
    return jsonify(signals)

@app.route('/api/search-signals')
def search_signals():
    """Search for signals by frequency, type, or other criteria"""
    frequency = request.args.get('frequency')
    signal_type = request.args.get('type')
    max_results = int(request.args.get('max_results', 50))
    
    try:
        # Get all detections
        signal_data = get_central_processor_data('detections')
        
        if not signal_data:
            return jsonify([])
        
        # Filter signals based on search criteria
        filtered_signals = []
        for signal in signal_data:
            match = True
            
            # Frequency filter (allow some tolerance)
            if frequency:
                try:
                    target_freq = float(frequency)
                    signal_freq = signal.get('frequency_mhz', 0)
                    freq_diff = abs(signal_freq - target_freq)
                    
                    # Allow 0.1 MHz tolerance
                    if freq_diff > 0.1:
                        match = False
                except ValueError:
                    match = False
            
            # Signal type filter
            if signal_type and signal_type.lower() != 'all':
                if signal.get('signal_type', '').lower() != signal_type.lower():
                    match = False
            
            if match:
                filtered_signals.append({
                    'id': signal.get('id'),
                    'frequency': signal.get('frequency_mhz', 0),
                    'signal_strength': signal.get('signal_strength_dbm', -100),
                    'lat': signal.get('lat', 35.4676),
                    'lng': signal.get('lng', -97.5164),
                    'detected_by': [signal.get('node_id', 'unknown')],
                    'timestamp': signal.get('timestamp', datetime.now().isoformat()),
                    'signal_type': signal.get('signal_type', 'Unknown'),
                    'confidence': signal.get('confidence', 0)
                })
        
        # Limit results
        return jsonify(filtered_signals[:max_results])
        
    except Exception as e:
        print(f"Error searching signals: {e}")
        return jsonify([])

@app.route('/api/system-status')
def get_system_status():
    """Get overall system status including development mode and buoy status"""
    dev_mode = config_manager.is_development_mode()
    
    # Check if we have real hardware (with caching)
    interfaces = get_cached_interfaces()
    has_real_buoys = bool(interfaces.get('sdr_devices')) and not dev_mode
    
    # Get GPS status
    gps_status = "unknown"
    if has_real_buoys:
        lat, lng, status = get_gps_coordinates()
        gps_status = status
    
    status = {
        'development_mode': dev_mode,
        'has_real_buoys': has_real_buoys,
        'sdr_devices_detected': len(interfaces.get('sdr_devices', [])),
        'gps_devices_detected': len(interfaces.get('gps_devices', [])),
        'gps_status': gps_status,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }
    
    return jsonify(status)

@app.route('/api/nodes')
def get_nodes():
    """Proxy route to get nodes from central processor"""
    try:
        nodes = get_central_processor_data('nodes')
        return jsonify(nodes)
    except Exception as e:
        return jsonify([]), 500

@app.route('/api/detections')
def get_detections():
    """Proxy route to get detections from central processor"""
    try:
        detections = get_central_processor_data('detections')
        return jsonify(detections)
    except Exception as e:
        return jsonify([]), 500

@app.route('/api/all-signals')
def get_all_signals():
    """Get all detected signals for the signal list modal"""
    try:
        signals = get_central_processor_data('detections')
        return jsonify(signals)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable or default to 7000
    port = int(os.environ.get('WEB_PORT', 7000))
    app.run(debug=True, host='0.0.0.0', port=port) 