#!/usr/bin/env python3
"""
Hardware detection test for Radio-Mapper system
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config_manager import ConfigManager

def test_hardware_detection():
    print("🔍 Radio-Mapper Hardware Detection Test")
    print("=" * 50)
    
    try:
        config_manager = ConfigManager()
        print("✅ ConfigManager loaded successfully")
        
        # Detect interfaces
        interfaces = config_manager.auto_detect_interfaces()
        print("✅ Interface detection completed")
        
        # Report findings
        sdr_devices = interfaces.get('sdr_devices', [])
        gps_devices = interfaces.get('gps_devices', [])
        
        print(f"\n📡 SDR Devices Found: {len(sdr_devices)}")
        for i, device in enumerate(sdr_devices):
            print(f"  Device {i+1}: {device}")
        
        print(f"\n🛰️  GPS Devices Found: {len(gps_devices)}")
        for i, device in enumerate(gps_devices):
            print(f"  Device {i+1}: {device}")
        
        # Check development mode
        dev_mode = config_manager.is_development_mode()
        print(f"\n🔧 Development Mode: {'ON' if dev_mode else 'OFF'}")
        
        # Recommendations
        print("\n💡 Recommendations:")
        if len(sdr_devices) == 0:
            print("  - No SDR devices detected. System will run in simulation mode.")
        elif len(sdr_devices) == 1:
            print(f"  - 1 SDR device detected. System should run 1 buoy, not 3!")
            print("  - Docker compose should be configured for single buoy operation.")
        else:
            print(f"  - {len(sdr_devices)} SDR devices detected. Multiple buoys can be configured.")
        
        if len(gps_devices) == 0:
            print("  - No GPS devices detected. System will use fallback coordinates.")
        else:
            print(f"  - {len(gps_devices)} GPS device(s) detected. Real GPS coordinates available.")
        
        return len(sdr_devices)
        
    except Exception as e:
        print(f"❌ Hardware detection failed: {e}")
        return -1

if __name__ == "__main__":
    sdr_count = test_hardware_detection()
    
    if sdr_count > 0:
        print(f"\n🎯 CONCLUSION: System should run {sdr_count} buoy(s), not 3!")
    elif sdr_count == 0:
        print(f"\n🎯 CONCLUSION: No SDR devices found. System running in simulation mode.")
    
    sys.exit(0 if sdr_count >= 0 else 1) 