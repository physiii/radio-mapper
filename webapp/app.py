
from flask import Flask, render_template, jsonify
import yaml
import os
from datetime import datetime, timedelta
import sys
import serial
import time
import re

# Add the parent directory to the path so we can import the config manager
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config_manager import ConfigManager

app = Flask(__name__)

# Load configuration using the actual config manager
config_manager = ConfigManager()
config = config_manager.config

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
    # Check if we're in development mode
    dev_mode = config_manager.is_development_mode()
    
    if dev_mode:
        # Return mock devices for development
        devices = [
            {'id': 1, 'name': 'OKC North', 'lat': 35.5200, 'lng': -97.5164, 'status': 'active', 'lastSeen': datetime.now().isoformat()},
            {'id': 2, 'name': 'OKC East', 'lat': 35.4676, 'lng': -97.4200, 'status': 'active', 'lastSeen': datetime.now().isoformat()},
            {'id': 3, 'name': 'OKC South', 'lat': 35.4100, 'lng': -97.5164, 'status': 'active', 'lastSeen': datetime.now().isoformat()},
            {'id': 4, 'name': 'OKC West', 'lat': 35.4676, 'lng': -97.6200, 'status': 'active', 'lastSeen': datetime.now().isoformat()}
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
    # Check if we're in development mode
    dev_mode = config_manager.is_development_mode()
    
    if dev_mode:
        # Return mock signals for development
        signals = [
            {
                'id': 1,
                'frequency': 144.5,  # MHz
                'signal_strength': -65,  # dBm
                'lat': 51.507,
                'lng': -0.095,
                'detected_by': [1, 2, 3],  # device IDs that detected this signal
                'timestamp': datetime.now().isoformat(),
                'signal_type': 'FM'
            },
            {
                'id': 2,
                'frequency': 433.92,  # MHz
                'signal_strength': -72,  # dBm
                'lat': 51.502,
                'lng': -0.105,
                'detected_by': [1, 3],
                'timestamp': datetime.now().isoformat(),
                'signal_type': 'Digital'
            },
            {
                'id': 3,
                'frequency': 868.3,  # MHz
                'signal_strength': -58,  # dBm
                'lat': 51.513,
                'lng': -0.088,
                'detected_by': [2, 3],
                'timestamp': datetime.now().isoformat(),
                'signal_type': 'LoRa'
            }
        ]
    else:
        # In production, this would return real signal data
        # For now, return empty list (no signals detected yet)
        signals = []
    
    return jsonify(signals)

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

if __name__ == '__main__':
    app.run(debug=True) 