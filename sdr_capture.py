#!/usr/bin/env python3
"""
Basic RTL-SDR IQ Data Capture Script
This script captures IQ data from an RTL-SDR device and saves it to a file.
"""

import subprocess
import time
import os
import sys
from datetime import datetime

def capture_iq_data(frequency_mhz, sample_rate=2048000, duration_seconds=10, output_file=None):
    """
    Capture IQ data from RTL-SDR device
    
    Args:
        frequency_mhz: Center frequency in MHz
        sample_rate: Sample rate in Hz (default 2048000)
        duration_seconds: Duration to capture in seconds
        output_file: Output filename (if None, auto-generated)
    """
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"iq_capture_{frequency_mhz}MHz_{timestamp}.bin"
    
    # Calculate number of samples
    num_samples = sample_rate * duration_seconds
    
    print(f"Starting IQ capture:")
    print(f"  Frequency: {frequency_mhz} MHz")
    print(f"  Sample Rate: {sample_rate} Hz")
    print(f"  Duration: {duration_seconds} seconds")
    print(f"  Output File: {output_file}")
    print(f"  Expected samples: {num_samples}")
    
    # Build rtl_sdr command
    cmd = [
        'rtl_sdr',
        '-f', str(int(frequency_mhz * 1e6)),  # Convert MHz to Hz
        '-s', str(sample_rate),
        '-n', str(num_samples),
        output_file
    ]
    
    try:
        # Run the capture command
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration_seconds + 10)
        
        if result.returncode == 0:
            print("Capture completed successfully!")
            
            # Check file size
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                expected_size = num_samples * 2  # 2 bytes per sample (I+Q)
                print(f"File size: {file_size} bytes (expected: {expected_size} bytes)")
                
                if file_size > 0:
                    print(f"Success! Captured data saved to {output_file}")
                    return True, output_file
                else:
                    print("Error: Output file is empty")
                    return False, None
            else:
                print("Error: Output file was not created")
                return False, None
        else:
            print(f"Error running rtl_sdr:")
            print(f"Return code: {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return False, None
            
    except subprocess.TimeoutExpired:
        print("Error: Capture timed out")
        return False, None
    except Exception as e:
        print(f"Error: {e}")
        return False, None

def test_sdr_device():
    """Test if SDR device is available and working"""
    print("Testing SDR device availability...")
    
    try:
        # Use rtl_test with a very short duration to avoid the E4000 error
        result = subprocess.run(['rtl_test', '-t'], capture_output=True, text=True, timeout=5)
        
        # Combine stdout and stderr for checking
        full_output = result.stdout + result.stderr
        print("RTL-SDR test output:")
        print(full_output[:500])  # Print first 500 chars to avoid spam
        
        if "Found" in full_output and "device" in full_output:
            print("✓ SDR device detected successfully")
            return True
        else:
            print("✗ No SDR device detected")
            return False
            
    except Exception as e:
        print(f"Error testing SDR device: {e}")
        return False

if __name__ == "__main__":
    print("=== RTL-SDR IQ Data Capture Tool ===")
    
    # Test device first
    if not test_sdr_device():
        print("Cannot proceed without a working SDR device")
        sys.exit(1)
    
    # Example captures
    print("\nStarting example captures...")
    
    # Capture FM radio band (around 100 MHz)
    success, filename = capture_iq_data(frequency_mhz=100.0, duration_seconds=5)
    if success:
        print(f"First capture successful: {filename}")
    
    time.sleep(1)
    
    # Capture 433 MHz ISM band
    success, filename = capture_iq_data(frequency_mhz=433.92, duration_seconds=5)
    if success:
        print(f"Second capture successful: {filename}")
    
    print("\nCapture test complete!") 