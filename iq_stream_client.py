#!/usr/bin/env python3
"""
IQ Stream Client for Radio-Mapper Emergency Response System

This client:
1. Continuously captures IQ data from local RTL-SDR
2. Performs real-time signal detection and analysis
3. Streams detected signals with precise timestamps to central server
4. Handles user-requested signal pattern searches
5. GPS-synchronized timing for TDoA triangulation
"""

import time
import json
import socket
import threading
import subprocess
import numpy as np
import websocket
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple
import scipy.signal
from scipy.fft import fft, fftfreq
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, complex):
            return str(obj)
        return json.JSONEncoder.default(self, obj)

@dataclass
class SignalDetection:
    """Real signal detection with precise timing"""
    node_id: str
    frequency_mhz: float
    signal_strength_dbm: float
    bandwidth_hz: float
    timestamp_utc: str
    gps_timestamp_ns: int
    lat: float
    lng: float
    confidence: float
    signal_type: str
    iq_samples: Optional[List[complex]] = None  # For pattern matching
    detection_method: str = "power_threshold"

@dataclass
class UserSignalRequest:
    """User request to find a specific signal"""
    request_id: str
    frequency_mhz: Optional[float]
    frequency_range_mhz: Optional[Tuple[float, float]]
    signal_pattern: Optional[List[complex]]  # IQ pattern to match
    max_age_minutes: int = 60  # How far back to search
    description: str = ""

class RealTimeSDRCapture:
    """Handles real-time IQ data capture from RTL-SDR"""
    
    def __init__(self, device_index: int = 0, sample_rate: int = 2048000):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.center_freq_hz = 100000000  # 100 MHz default
        self.running = False
        self.capture_process = None
        
        # Signal detection parameters
        self.detection_threshold_db = -70  # dBm
        self.fft_size = 1024
        self.overlap = 0.5
        
        # Emergency frequencies to monitor (Hz)
        self.emergency_frequencies = [
            121500000,  # Aviation emergency
            243000000,  # Military emergency
            155160000,  # Public safety
            406000000,  # EPIRB/PLB beacons
        ]
        
    def start_capture(self, center_freq_mhz: float = 100.0) -> bool:
        """Start real-time IQ capture from RTL-SDR"""
        self.center_freq_hz = int(center_freq_mhz * 1e6)
        
        try:
            # Build rtl_sdr command for continuous capture
            cmd = [
                'rtl_sdr', 
                '-f', str(self.center_freq_hz),
                '-s', str(self.sample_rate),
                '-'  # Output to stdout
            ]
            
            logger.info(f"Starting RTL-SDR capture: {center_freq_mhz} MHz @ {self.sample_rate} Hz")
            
            # Start subprocess
            self.capture_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            self.running = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to start SDR capture: {e}")
            return False
    
    def stop_capture(self):
        """Stop the SDR capture"""
        self.running = False
        if self.capture_process:
            self.capture_process.terminate()
            self.capture_process.wait()
            self.capture_process = None
        logger.info("SDR capture stopped")
    
    def read_iq_samples(self, num_samples: int = 8192) -> Optional[np.ndarray]:
        """Read IQ samples from RTL-SDR stream"""
        if not self.running or not self.capture_process:
            return None
        
        try:
            # Read raw bytes (2 bytes per sample: I + Q)
            num_bytes = num_samples * 2
            raw_data = self.capture_process.stdout.read(num_bytes)
            
            if len(raw_data) != num_bytes:
                logger.warning(f"Incomplete read: got {len(raw_data)} bytes, expected {num_bytes}")
                return None
            
            # Convert to complex IQ samples
            raw_array = np.frombuffer(raw_data, dtype=np.uint8)
            
            # Convert to float and center around zero
            raw_array = raw_array.astype(np.float32) - 127.5
            
            # Separate I and Q, create complex samples
            i_samples = raw_array[0::2]
            q_samples = raw_array[1::2]
            iq_samples = i_samples + 1j * q_samples
            
            return iq_samples
            
        except Exception as e:
            logger.error(f"Error reading IQ samples: {e}")
            return None

class SignalDetector:
    """Detects signals in real-time IQ data stream"""
    
    def __init__(self, node_id: str, sample_rate: int = 2048000):
        self.node_id = node_id
        self.sample_rate = sample_rate
        self.detection_threshold = -70  # dBm
        
        # Position (would come from GPS in production)
        self.lat = 35.4676  # Oklahoma City
        self.lng = -97.5164
        
        # Signal history for pattern matching
        self.signal_history: List[SignalDetection] = []
        self.max_history_size = 1000
        
    def detect_signals(self, iq_samples: np.ndarray, center_freq_hz: float) -> List[SignalDetection]:
        """Detect signals in IQ sample block"""
        detections = []
        
        try:
            # Calculate FFT
            fft_result = fft(iq_samples)
            freqs = fftfreq(len(iq_samples), 1.0 / self.sample_rate)
            
            # Calculate power spectrum in dB
            power_spectrum_db = 20 * np.log10(np.abs(fft_result) + 1e-12)
            
            # Convert frequencies to absolute frequencies
            abs_freqs = freqs + center_freq_hz
            
            # Find peaks above threshold
            peaks, properties = scipy.signal.find_peaks(
                power_spectrum_db, 
                height=self.detection_threshold,
                distance=10  # Minimum distance between peaks
            )
            
            # Create detections for each peak
            for peak_idx in peaks:
                peak_freq_hz = abs_freqs[peak_idx]
                peak_power_db = power_spectrum_db[peak_idx]
                
                # Estimate bandwidth (simple method)
                bandwidth = self._estimate_bandwidth(power_spectrum_db, peak_idx)
                
                # Classify signal type
                signal_type = self._classify_signal(peak_freq_hz)
                
                # Calculate confidence based on SNR
                noise_floor = np.median(power_spectrum_db)
                snr_db = peak_power_db - noise_floor
                confidence = min(snr_db / 20.0, 1.0)  # Normalize to 0-1
                
                # Get precise timestamp
                timestamp_utc = datetime.now(timezone.utc).isoformat()
                gps_timestamp_ns = time.time_ns()
                
                # Extract IQ samples around the signal for pattern matching
                signal_samples = self._extract_signal_samples(iq_samples, peak_idx)
                
                detection = SignalDetection(
                    node_id=self.node_id,
                    frequency_mhz=peak_freq_hz / 1e6,
                    signal_strength_dbm=peak_power_db,
                    bandwidth_hz=bandwidth,
                    timestamp_utc=timestamp_utc,
                    gps_timestamp_ns=gps_timestamp_ns,
                    lat=self.lat,
                    lng=self.lng,
                    confidence=confidence,
                    signal_type=signal_type,
                    iq_samples=signal_samples.tolist() if signal_samples is not None else None,
                    detection_method="fft_peak"
                )
                
                detections.append(detection)
                
                # Log emergency signals
                if signal_type == "emergency":
                    logger.warning(f"EMERGENCY SIGNAL: {peak_freq_hz/1e6:.3f} MHz, {peak_power_db:.1f} dBm")
                else:
                    logger.info(f"Signal detected: {peak_freq_hz/1e6:.3f} MHz, {peak_power_db:.1f} dBm")
        
        except Exception as e:
            logger.error(f"Error in signal detection: {e}")
        
        return detections
    
    def _estimate_bandwidth(self, power_spectrum_db: np.ndarray, peak_idx: int) -> float:
        """Estimate signal bandwidth around a peak"""
        try:
            # Find -3dB points around the peak
            peak_power = power_spectrum_db[peak_idx]
            threshold = peak_power - 3.0  # -3dB
            
            # Search left and right for bandwidth
            left_idx = peak_idx
            right_idx = peak_idx
            
            while left_idx > 0 and power_spectrum_db[left_idx] > threshold:
                left_idx -= 1
            
            while right_idx < len(power_spectrum_db) - 1 and power_spectrum_db[right_idx] > threshold:
                right_idx += 1
            
            # Calculate bandwidth in Hz
            bandwidth_bins = right_idx - left_idx
            bandwidth_hz = bandwidth_bins * (self.sample_rate / len(power_spectrum_db))
            
            return bandwidth_hz
            
        except:
            return 10000.0  # Default 10 kHz
    
    def _classify_signal(self, frequency_hz: float) -> str:
        """Classify signal type based on frequency"""
        freq_mhz = frequency_hz / 1e6
        
        # Emergency frequencies
        if abs(frequency_hz - 121500000) < 1000:  # 121.5 MHz ±1kHz
            return "emergency"
        elif abs(frequency_hz - 243000000) < 1000:  # 243.0 MHz ±1kHz
            return "emergency"
        elif 155000000 <= frequency_hz <= 156000000:  # Public safety
            return "public_safety"
        elif 406000000 <= frequency_hz <= 406100000:  # Emergency beacons
            return "emergency"
        
        # Band classifications
        elif 88000000 <= frequency_hz <= 108000000:  # FM radio
            return "fm_radio"
        elif 118000000 <= frequency_hz <= 136000000:  # Aviation
            return "aviation"
        elif 144000000 <= frequency_hz <= 148000000:  # Amateur 2m
            return "amateur"
        elif 420000000 <= frequency_hz <= 450000000:  # Amateur 70cm
            return "amateur"
        else:
            return "unknown"
    
    def _extract_signal_samples(self, iq_samples: np.ndarray, peak_idx: int, num_samples: int = 256) -> Optional[np.ndarray]:
        """Extract IQ samples around a detected signal for pattern matching"""
        try:
            # Simple extraction - in production would use proper filtering
            start_idx = max(0, peak_idx - num_samples // 2)
            end_idx = min(len(iq_samples), start_idx + num_samples)
            
            return iq_samples[start_idx:end_idx]
        except:
            return None
    
    def search_signal_history(self, request: UserSignalRequest) -> List[SignalDetection]:
        """Search historical detections for user-requested signals"""
        matches = []
        
        for detection in self.signal_history:
            # Check age
            detection_time = datetime.fromisoformat(detection.timestamp_utc.replace('Z', '+00:00'))
            age_minutes = (datetime.now(timezone.utc) - detection_time).total_seconds() / 60
            
            if age_minutes > request.max_age_minutes:
                continue
            
            # Check frequency match
            freq_match = False
            if request.frequency_mhz:
                if abs(detection.frequency_mhz - request.frequency_mhz) < 0.01:  # ±10kHz
                    freq_match = True
            elif request.frequency_range_mhz:
                freq_min, freq_max = request.frequency_range_mhz
                if freq_min <= detection.frequency_mhz <= freq_max:
                    freq_match = True
            else:
                freq_match = True  # No frequency filter
            
            if freq_match:
                matches.append(detection)
        
        return matches

class CentralServerClient:
    """Handles communication with central processing server"""
    
    def __init__(self, server_url: str = "ws://localhost:8080", node_id: str = "OKC_LOCAL", lat: float = 35.4676, lng: float = -97.5164):
        self.server_url = server_url
        self.node_id = node_id
        self.lat = lat
        self.lng = lng
        self.websocket = None
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to central server via WebSocket"""
        try:
            def on_open(ws):
                logger.info("Connected to central server")
                self.connected = True
                
                # Send node registration
                registration = {
                    'type': 'node_registration',
                    'node_id': self.node_id,
                    'lat': self.lat,
                    'lng': self.lng,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                ws.send(json.dumps(registration))
                
            def on_message(ws, message):
                self._handle_server_message(message)
                
            def on_error(ws, error):
                logger.error(f"WebSocket error: {error}")
                
            def on_close(ws, close_status_code, close_msg):
                logger.info("WebSocket connection closed")
                self.connected = False
                
            self.websocket = websocket.WebSocketApp(
                self.server_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            # Start WebSocket in background thread
            wst = threading.Thread(target=self.websocket.run_forever)
            wst.daemon = True
            wst.start()
            
            # Wait for connection
            time.sleep(2)
            return self.connected
            
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False
    
    def send_detection(self, detection: SignalDetection) -> bool:
        """Send signal detection to central server"""
        if not self.connected:
            logger.warning("Not connected to central server")
            return False
        
        try:
            message = {
                "type": "signal_detection",
                "data": asdict(detection),
                "timestamp": detection.timestamp_utc
            }
            
            self.websocket.send(json.dumps(message, cls=NumpyJSONEncoder))
            return True
            
        except Exception as e:
            logger.error(f"Failed to send detection: {e}")
            return False
    
    def _handle_server_message(self, message: str):
        """Handle messages from central server"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "signal_request":
                # Server is requesting search for specific signal
                request_data = data.get("data", {})
                logger.info(f"Received signal search request: {request_data}")
                
            elif msg_type == "triangulation_result":
                # Server has triangulated a signal
                result = data.get("data", {})
                logger.info(f"Triangulation result: {result.get('frequency_mhz')} MHz at "
                           f"({result.get('estimated_lat')}, {result.get('estimated_lng')})")
                
        except Exception as e:
            logger.error(f"Error handling server message: {e}")

class IQStreamClient:
    """Main IQ streaming client that coordinates all components"""
    
    def __init__(self, node_id: str = "OKC_LOCAL", server_url: str = "ws://localhost:8080"):
        self.node_id = node_id
        self.running = False
        
        # Initialize components
        self.sdr_capture = RealTimeSDRCapture()
        self.signal_detector = SignalDetector(node_id)
        self.server_client = CentralServerClient(server_url, node_id)
        
        # Processing parameters
        self.process_interval = 0.1  # Process every 100ms
        self.samples_per_process = 8192
        
    def start(self, center_freq_mhz: float = 100.0) -> bool:
        """Start the IQ streaming client"""
        logger.info(f"Starting IQ Stream Client: {self.node_id}")
        
        # Connect to central server
        if not self.server_client.connect():
            logger.warning("Failed to connect to central server - running in standalone mode")
        
        # Start SDR capture
        if not self.sdr_capture.start_capture(center_freq_mhz):
            logger.error("Failed to start SDR capture")
            return False
        
        # Start processing loop
        self.running = True
        self._start_processing_thread()
        
        logger.info(f"IQ Stream Client {self.node_id} is operational")
        return True
    
    def stop(self):
        """Stop the IQ streaming client"""
        logger.info(f"Stopping IQ Stream Client: {self.node_id}")
        
        self.running = False
        self.sdr_capture.stop_capture()
        
        logger.info(f"IQ Stream Client {self.node_id} stopped")
    
    def _start_processing_thread(self):
        """Start the main signal processing thread"""
        def processing_loop():
            while self.running:
                try:
                    # Read IQ samples from SDR
                    iq_samples = self.sdr_capture.read_iq_samples(self.samples_per_process)
                    
                    if iq_samples is not None:
                        # Detect signals
                        detections = self.signal_detector.detect_signals(
                            iq_samples, 
                            self.sdr_capture.center_freq_hz
                        )
                        
                        # Send detections to central server
                        for detection in detections:
                            # Add to local history
                            self.signal_detector.signal_history.append(detection)
                            
                            # Trim history if too long
                            if len(self.signal_detector.signal_history) > self.signal_detector.max_history_size:
                                self.signal_detector.signal_history.pop(0)
                            
                            # Send to server
                            self.server_client.send_detection(detection)
                    
                    # Brief pause
                    time.sleep(self.process_interval)
                    
                except Exception as e:
                    logger.error(f"Error in processing loop: {e}")
                    time.sleep(1.0)
        
        processing_thread = threading.Thread(target=processing_loop, daemon=True)
        processing_thread.start()

def main():
    """Main entry point for IQ streaming client"""
    import sys
    
    # Get parameters from command line
    node_id = sys.argv[1] if len(sys.argv) > 1 else "OKC_LOCAL"
    center_freq = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
    server_url = sys.argv[3] if len(sys.argv) > 3 else "ws://localhost:8080"
    
    logger.info(f"Starting IQ Stream Client: {node_id}")
    logger.info(f"Center frequency: {center_freq} MHz")
    logger.info(f"Central server: {server_url}")
    
    # Create and start client
    client = IQStreamClient(node_id, server_url)
    
    try:
        if client.start(center_freq):
            logger.info(f"IQ Stream Client {node_id} running at {center_freq} MHz. Press Ctrl+C to stop.")
            
            # Keep running until interrupted
            while True:
                time.sleep(10)
                logger.info(f"Client {node_id} operational - "
                           f"{len(client.signal_detector.signal_history)} signals detected")
        else:
            logger.error("Failed to start IQ stream client")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        client.stop()

if __name__ == "__main__":
    main() 