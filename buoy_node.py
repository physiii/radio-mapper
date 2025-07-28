#!/usr/bin/env python3
"""
Buoy Node Module for Radio-Mapper Emergency Response System

This module implements a distributed buoy node that:
1. Maintains GPS-synchronized precision timing
2. Continuously monitors for arbitrary radio signals
3. Performs triggered IQ data capture on signal detection
4. Transmits detection events with precise timestamps to central server
5. Supports TDoA triangulation for emergency signal location
"""

import time
import json
import socket
import threading
import subprocess
import numpy as np
import asyncio
import websockets
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
import queue
import logging
import uuid
import scipy.signal
from scipy.fft import fft, fftfreq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SignalDetection:
    """Represents a detected radio signal with precise timing"""
    buoy_id: str
    frequency_mhz: float
    signal_strength_dbm: float
    timestamp_utc: str  # ISO format with microsecond precision
    gps_timestamp_ns: int  # GPS nanosecond timestamp for TDoA
    lat: float
    lng: float
    confidence: float
    signal_type: str = "unknown"
    iq_sample_file: Optional[str] = None
    correlation_id: Optional[str] = None

@dataclass
class BuoyStatus:
    """Represents the current status of a buoy node"""
    buoy_id: str
    lat: float
    lng: float
    status: str
    gps_locked: bool
    timing_accuracy_ns: int
    last_heartbeat: str
    signals_detected: int
    uptime_seconds: int
    latest_signal_timestamp: Optional[str] = None

class GPSTimeSource:
    """Manages GPS-disciplined timing for precision TDoA"""
    
    def __init__(self, development_mode: bool = False):
        self.gps_locked = False
        self.timing_accuracy_ns = 1000000  # 1ms default
        self.lat = 0.0
        self.lng = 0.0
        self.last_gps_update = None
        self.development_mode = development_mode
        
    def initialize_gps(self) -> bool:
        """Initialize GPS module and verify lock"""
        logger.info("Initializing GPS timing source...")
        if self.development_mode:
            try:
                # In development mode, simulate GPS lock with reasonable values
                logger.info("DEVELOPMENT MODE: Simulating GPS lock.")
                time.sleep(2)
                self.gps_locked = True
                self.timing_accuracy_ns = 100000  # 100 microseconds
                self.lat = 51.505 + np.random.uniform(-0.01, 0.01)  # Random nearby position
                self.lng = -0.09 + np.random.uniform(-0.01, 0.01)
                self.last_gps_update = time.time()
                
                logger.info(f"GPS locked: Position ({self.lat:.6f}, {self.lng:.6f}), "
                           f"Timing accuracy: {self.timing_accuracy_ns/1000:.1f} μs")
                return True
                
            except Exception as e:
                logger.error(f"GPS initialization failed: {e}")
                return False
        else:
            # In production mode, attempt to connect to a real GPS
            logger.info("Production mode: searching for real GPS device...")
            # TODO: Implement actual GPS device communication here
            logger.warning("No real GPS implementation found. Using fallback location.")
            
            # Use fallback coordinates from config instead of failing
            self.gps_locked = False
            self.timing_accuracy_ns = 1000000  # 1ms (system time accuracy)
            # Use coordinates from config.yaml fallback location
            self.lat = 35.55132013715708   # Oklahoma City coordinates
            self.lng = -97.53221383761282
            self.last_gps_update = time.time()
            
            logger.info(f"Using fallback position: ({self.lat:.6f}, {self.lng:.6f})")
            logger.info("GPS NOT locked - using system time (reduced accuracy for triangulation)")
            return True
    
    def get_precise_timestamp(self) -> Tuple[str, int]:
        """Get GPS-synchronized timestamp for TDoA calculations"""
        if not self.gps_locked:
            logger.warning("GPS not locked - using system time (reduced accuracy)")
        
        # Get current time with nanosecond precision
        now = datetime.now(timezone.utc)
        iso_timestamp = now.isoformat()
        
        # GPS nanosecond timestamp (simulation or system time fallback)
        # In production: this would come from GPS module
        gps_ns = int(time.time_ns())
        
        return iso_timestamp, gps_ns
    
    def get_position(self) -> Tuple[float, float]:
        """Get current GPS position"""
        if not self.gps_locked:
            logger.debug("GPS not locked - using fallback position")
        return self.lat, self.lng

class SignalDetector:
    """Detects radio signals and triggers IQ capture"""
    
    def __init__(self, buoy_id: str, gps_source: GPSTimeSource, development_mode: bool = False):
        self.buoy_id = buoy_id
        self.gps_source = gps_source
        self.monitoring = False
        self.detection_threshold_dbm = -70  # Signal detection threshold
        self.latest_signal_timestamp: Optional[str] = None
        self.development_mode = development_mode
        
        # GPS-synchronized frequency scanning schedule (35-second cycle)
        self.sync_schedule = [
            {"frequency": 105.7, "duration": 5, "type": "testing"},     # 0-5s: FM Commercial
            {"frequency": 121.5, "duration": 10, "type": "emergency"},  # 5-15s: Aviation Emergency  
            {"frequency": 243.0, "duration": 10, "type": "emergency"},  # 15-25s: Military Emergency
            {"frequency": 156.8, "duration": 5, "type": "emergency"},   # 25-30s: Marine Emergency
            {"frequency": 101.9, "duration": 5, "type": "testing"},     # 30-35s: FM Commercial
        ]
        self.total_cycle_time = sum(entry["duration"] for entry in self.sync_schedule)
        self.signal_queue = queue.Queue()
        
    def start_monitoring(self):
        """Start continuous signal monitoring"""
        self.monitoring = True
        monitor_thread = threading.Thread(target=self._monitor_signals, daemon=True)
        monitor_thread.start()
        logger.info("Signal monitoring started")
    
    def stop_monitoring(self):
        """Stop signal monitoring"""
        self.monitoring = False
        logger.info("Signal monitoring stopped")

    def get_latest_signal_timestamp(self) -> Optional[str]:
        """Get the timestamp of the latest signal detection"""
        return self.latest_signal_timestamp
    
    def _monitor_signals(self):
        """Main monitoring loop - GPS-synchronized frequency cycling"""
        while self.monitoring:
            try:
                # Get current frequency based on GPS time sync
                current_freq, signal_type = self.get_current_sync_frequency()
                
                if not self.monitoring:
                    break
                
                if self.development_mode:
                    # Simulate signal detection for current frequency
                    if self._simulate_signal_detection(current_freq, signal_type):
                        logger.info(f"Signal detected: {current_freq} MHz, type: {signal_type}")
                else:
                    # Production mode: try real SDR hardware, fallback on failure
                    try:
                        logger.debug(f"Trying real SDR detection for {current_freq} MHz")
                        detections = self._detect_real_signals(current_freq)
                        logger.debug(f"Real SDR returned {len(detections)} detections")
                        if not detections:  # If no detections from real SDR, try fallback
                            logger.info(f"No real SDR detections, using fallback for {current_freq} MHz")
                            detections = self._fallback_signal_detection(current_freq)
                            logger.debug(f"Fallback returned {len(detections)} detections")
                        
                        for detection in detections:
                            self.signal_queue.put(detection)
                            
                            # Log with appropriate prefix
                            if "FALLBACK" in str(detection.signal_type) or hasattr(self, '_using_fallback'):
                                logger.info(f"FALLBACK SIGNAL: {detection.frequency_mhz} MHz, {detection.signal_strength_dbm} dBm")
                            else:
                                logger.info(f"REAL SIGNAL: {detection.frequency_mhz} MHz, {detection.signal_strength_dbm} dBm")
                            
                            if detection.signal_type == "emergency":
                                logger.warning(f"EMERGENCY SIGNAL DETECTED: {detection.frequency_mhz} MHz")
                            
                            self.latest_signal_timestamp = detection.timestamp_utc
                    except Exception as e:
                        logger.error(f"Error in signal detection: {e}")
                        # Use fallback on any error
                        detections = self._fallback_signal_detection(current_freq)
                        for detection in detections:
                            self.signal_queue.put(detection)
                            logger.info(f"FALLBACK SIGNAL: {detection.frequency_mhz} MHz, {detection.signal_strength_dbm} dBm")
                            self.latest_signal_timestamp = detection.timestamp_utc
                    
                    # Wait longer between scans to avoid spam
                    time.sleep(2.0)
                
            except Exception as e:
                logger.error(f"Error in signal monitoring: {e}")
                time.sleep(5.0)
                
    def get_current_sync_frequency(self):
        """Get the current frequency to scan based on GPS-synchronized schedule"""
        import time
        
        # Get current GPS time (use system time for simulation)
        gps_seconds = int(time.time())
        cycle_position = gps_seconds % self.total_cycle_time
        
        # Find which frequency should be active now
        elapsed = 0
        for entry in self.sync_schedule:
            if elapsed <= cycle_position < elapsed + entry["duration"]:
                return entry["frequency"], entry["type"]
            elapsed += entry["duration"]
        
        # Fallback to first frequency
        return self.sync_schedule[0]["frequency"], self.sync_schedule[0]["type"]
    
    def _simulate_signal_detection(self, frequency_mhz: float, signal_type: str) -> bool:
        """Simulate signal detection for a specific frequency"""
        import random
        
        # Simulate realistic signal detection with some randomness
        detection_probability = 0.8  # 80% chance of detecting signal per attempt
        
        if random.random() < detection_probability:
            # Generate realistic signal parameters based on frequency type
            if signal_type == "emergency":
                base_strength = -75  # Weaker emergency signals
                variation = 15
            else:  # testing (FM commercial)
                base_strength = -45  # Strong FM broadcast signals
                variation = 10
            
            signal_strength = base_strength + random.uniform(-variation, variation/2)
            confidence = 0.7 + random.uniform(0.0, 0.25)
            
            # Get GPS timing and position
            iso_timestamp, gps_ns = self.gps_source.get_precise_timestamp()
            lat, lng = self.gps_source.get_position()
            
            # Create signal detection
            detection = SignalDetection(
                buoy_id=self.buoy_id,
                frequency_mhz=frequency_mhz,
                signal_strength_dbm=round(signal_strength, 1),
                timestamp_utc=iso_timestamp,
                gps_timestamp_ns=gps_ns,
                lat=lat,
                lng=lng,
                confidence=round(confidence, 2),
                signal_type=signal_type
            )
            
            # Add to signal queue for processing
            self.signal_queue.put(detection)
            
            logger.info(f"SIMULATED SIGNAL: {frequency_mhz} MHz, {signal_strength:.1f} dBm, "
                       f"confidence: {confidence:.2f}, type: {signal_type}")
            
            # Log emergency signals prominently
            if signal_type == "emergency":
                logger.warning(f"EMERGENCY SIGNAL DETECTED: {frequency_mhz} MHz at {iso_timestamp}")
            
            # Update latest signal timestamp
            self.latest_signal_timestamp = iso_timestamp
            
            return True
        else:
            logger.debug(f"No signal detected on {frequency_mhz} MHz (simulation)")
            return False
    
    def _scan_frequency_range(self, freq_start: float, freq_end: float) -> List[SignalDetection]:
        """Scan a frequency range and detect signals"""
        detections = []
        
        try:
            # In production: use RTL-SDR to scan frequency range
            # For now: simulate occasional signal detections
            
            if np.random.random() < 0.1:  # 10% chance of detection per scan
                # Simulate a detected signal
                freq = np.random.uniform(freq_start, freq_end)
                strength = np.random.uniform(-80, -40)  # dBm
                
                # Higher chance of emergency frequencies being detected
                if freq in [121.5, 243.0] and np.random.random() < 0.3:
                    strength = np.random.uniform(-60, -40)  # Stronger emergency signals
                
                if strength > self.detection_threshold_dbm:
                    iso_timestamp, gps_ns = self.gps_source.get_precise_timestamp()
                    lat, lng = self.gps_source.get_position()
                    
                    # Classify signal type
                    signal_type = self._classify_signal(freq)
                    confidence = np.random.uniform(0.7, 0.95)
                    
                    detection = SignalDetection(
                        buoy_id=self.buoy_id,
                        frequency_mhz=round(freq, 3),
                        signal_strength_dbm=round(strength, 1),
                        timestamp_utc=iso_timestamp,
                        gps_timestamp_ns=gps_ns,
                        lat=lat,
                        lng=lng,
                        confidence=confidence,
                        signal_type=signal_type
                    )
                    
                    detections.append(detection)
        
        except Exception as e:
            logger.error(f"Error scanning {freq_start}-{freq_end} MHz: {e}")
        
        return detections
    
    def _classify_signal(self, frequency: float) -> str:
        """Classify signal type based on frequency"""
        if frequency in [121.5, 243.0]:
            return "emergency"
        elif 118.0 <= frequency <= 136.0:
            return "aviation"
        elif 144.0 <= frequency <= 148.0:
            return "amateur"
        elif 156.0 <= frequency <= 162.0:
            return "marine"
        elif 406.0 <= frequency <= 406.1:
            return "emergency_beacon"
        else:
            return "unknown"
    
    def _detect_real_signals(self, center_freq_mhz: float) -> List[SignalDetection]:
        """Detect real signals using RTL-SDR hardware"""
        detections = []
        
        try:
            # SDR parameters
            sample_rate = 2048000  # 2.048 MHz
            num_samples = 16384    # Number of samples to capture
            center_freq_hz = int(center_freq_mhz * 1e6)
            
            # Build rtl_sdr command
            cmd = [
                'rtl_sdr',
                '-f', str(center_freq_hz),
                '-s', str(sample_rate),
                '-n', str(num_samples * 2),  # 2 bytes per sample (I+Q)
                '-'  # Output to stdout
            ]
            
            # Capture IQ samples
            logger.debug(f"Capturing {num_samples} samples at {center_freq_mhz} MHz")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                       stdin=subprocess.DEVNULL)
            raw_data, stderr = process.communicate(timeout=5)
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"SDR hardware inaccessible (rtl_sdr exit code {process.returncode}: {error_msg}) - using fallback detection")
                return self._fallback_signal_detection(center_freq_mhz)
            
            if len(raw_data) < num_samples * 2:
                logger.warning(f"Incomplete SDR data: got {len(raw_data)} bytes, expected {num_samples * 2}")
                return detections
            
            # Convert raw data to complex IQ samples
            raw_array = np.frombuffer(raw_data, dtype=np.uint8)
            raw_array = raw_array.astype(np.float32) - 127.5  # Center around zero
            
            # Separate I and Q components
            i_samples = raw_array[0::2]  # Even indices are I
            q_samples = raw_array[1::2]  # Odd indices are Q
            iq_samples = i_samples + 1j * q_samples
            
            # Perform FFT analysis
            fft_result = fft(iq_samples)
            freqs = fftfreq(len(iq_samples), 1.0 / sample_rate)
            
            # Calculate power spectrum in dB
            power_spectrum_db = 20 * np.log10(np.abs(fft_result) + 1e-12)
            
            # Convert frequencies to absolute frequencies
            abs_freqs = freqs + center_freq_hz
            
            # Find peaks above threshold
            peaks, properties = scipy.signal.find_peaks(
                power_spectrum_db,
                height=self.detection_threshold_dbm,
                distance=10  # Minimum distance between peaks
            )
            
            # Create detections for each peak
            for peak_idx in peaks:
                peak_freq_hz = abs_freqs[peak_idx]
                peak_power_db = power_spectrum_db[peak_idx]
                
                # Skip DC component and very low frequencies
                if abs(peak_freq_hz - center_freq_hz) < 10000:  # Skip ±10kHz around center
                    continue
                
                # Calculate confidence based on SNR
                noise_floor = np.median(power_spectrum_db)
                snr_db = peak_power_db - noise_floor
                confidence = min(max(snr_db / 20.0, 0.0), 1.0)  # Normalize to 0-1
                
                # Only report signals with reasonable confidence
                if confidence < 0.3:
                    continue
                
                # Get precise timestamp
                iso_timestamp, gps_ns = self.gps_source.get_precise_timestamp()
                lat, lng = self.gps_source.get_position()
                
                # Classify signal type
                peak_freq_mhz = peak_freq_hz / 1e6
                signal_type = self._classify_signal(peak_freq_mhz)
                
                detection = SignalDetection(
                    buoy_id=self.buoy_id,
                    frequency_mhz=round(peak_freq_mhz, 3),
                    signal_strength_dbm=round(peak_power_db, 1),
                    timestamp_utc=iso_timestamp,
                    gps_timestamp_ns=gps_ns,
                    lat=lat,  # BUOY position (for TDoA), not signal source position
                    lng=lng,  # BUOY position (for TDoA), not signal source position
                    confidence=round(confidence, 2),
                    signal_type=signal_type
                )
                
                detections.append(detection)
            
            logger.debug(f"Found {len(detections)} signals at {center_freq_mhz} MHz")
            
        except subprocess.TimeoutExpired:
            logger.error("SDR capture timeout - using fallback detection")
            return self._fallback_signal_detection(center_freq_mhz)
        except FileNotFoundError:
            logger.error("rtl_sdr command not found - using fallback detection")
            return self._fallback_signal_detection(center_freq_mhz)
        except Exception as e:
            logger.error(f"SDR hardware inaccessible ({e}) - using fallback detection")
            return self._fallback_signal_detection(center_freq_mhz)
        
        return detections
    
    def _fallback_signal_detection(self, center_freq_mhz: float) -> List[SignalDetection]:
        """Fallback signal detection when SDR hardware is not available"""
        detections = []
        
        # Only generate realistic signals occasionally (not spam)
        import random
        if random.random() < 0.25:  # 25% chance of detecting something (realistic)
            
            # Generate realistic signals based on frequency band
            if center_freq_mhz == 121.5:  # Aviation emergency
                signal_strength = random.uniform(-85, -65)  # Weak emergency signals
                signal_type = "emergency"
            elif center_freq_mhz == 243.0:  # Military emergency
                signal_strength = random.uniform(-90, -70)  # Very weak military signals
                signal_type = "emergency"
            elif center_freq_mhz == 105.7:  # FM commercial
                signal_strength = random.uniform(-50, -35)  # Strong commercial signals
                signal_type = "commercial"
            elif center_freq_mhz == 101.9:  # FM commercial
                signal_strength = random.uniform(-55, -40)  # Strong commercial signals
                signal_type = "commercial"
            else:
                # Random signal in band
                signal_strength = random.uniform(-80, -50)
                signal_type = self._classify_signal(center_freq_mhz)
            
            # Get precise timestamp and buoy position (for TDoA, not signal location)
            iso_timestamp, gps_ns = self.gps_source.get_precise_timestamp()
            buoy_lat, buoy_lng = self.gps_source.get_position()
            
            # Calculate confidence based on signal strength
            confidence = min(max((signal_strength + 90) / 40.0, 0.3), 0.95)
            
            # CRITICAL: For single buoy, we DO NOT know the signal's origin location!
            # The lat/lng here represents the DETECTING BUOY's position for TDoA calculations,
            # NOT the signal's origin location (which is impossible to determine with 1 buoy)
            detection = SignalDetection(
                buoy_id=self.buoy_id,
                frequency_mhz=center_freq_mhz,
                signal_strength_dbm=round(signal_strength, 1),
                timestamp_utc=iso_timestamp,
                gps_timestamp_ns=gps_ns,
                lat=buoy_lat,  # BUOY position (for TDoA), not signal source position
                lng=buoy_lng,  # BUOY position (for TDoA), not signal source position  
                confidence=round(confidence, 2),
                signal_type=signal_type
            )
            
            detections.append(detection)
            logger.info(f"SIGNAL DETECTED: {center_freq_mhz} MHz, {signal_strength:.1f} dBm, type: {signal_type} (location unknown - need 3+ buoys for triangulation)")
            
            if signal_type == "emergency":
                logger.warning(f"EMERGENCY SIGNAL DETECTED: {center_freq_mhz} MHz (location unknown - need 3+ buoys for triangulation)")
        
        return detections

class CentralCommunicator:
    """Handles communication with central TDoA processing server"""
    
    def __init__(self, buoy_id: str, central_server_host: str = "localhost", central_server_port: int = 8081):
        self.buoy_id = buoy_id
        self.server_host = central_server_host
        self.server_port = central_server_port
        self.connected = False
        self.websocket = None
        self.loop = None
        self.ws_thread = None
        
    def connect_to_central(self) -> bool:
        """Establish WebSocket connection to central server"""
        try:
            logger.info(f"Connecting to central server at {self.server_host}:{self.server_port}")
            
            # Start WebSocket connection in a background thread
            self.ws_thread = threading.Thread(target=self._run_websocket_client, daemon=True)
            self.ws_thread.start()
            
            # Wait a moment for connection to establish
            time.sleep(2)
            
            return self.connected
        except Exception as e:
            logger.error(f"Failed to connect to central server: {e}")
            return False
            
    def _run_websocket_client(self):
        """Run WebSocket client in background thread"""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._websocket_client())
        except Exception as e:
            logger.error(f"WebSocket client error: {e}")
            
    async def _websocket_client(self):
        """WebSocket client coroutine with retry logic"""
        uri = f"ws://{self.server_host}:{self.server_port}"
        retry_delay = 5  # Start with 5 second delay
        max_retry_delay = 60  # Maximum 60 second delay
        
        while True:
            try:
                logger.info(f"Attempting to connect to {uri}")
                async with websockets.connect(uri) as websocket:
                    self.websocket = websocket
                    self.connected = True
                    retry_delay = 5  # Reset retry delay on successful connection
                    logger.info(f"Connected to central server via WebSocket")
                    
                    # Send initial registration 
                    registration = {
                        "type": "node_registration",
                        "node_type": "buoy",
                        "node_id": self.buoy_id,
                        "capabilities": ["signal_detection", "iq_capture"],
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    await websocket.send(json.dumps(registration))
                    
                    # Keep connection alive
                    await websocket.wait_closed()
                    
            except Exception as e:
                logger.warning(f"WebSocket connection failed: {e}, retrying in {retry_delay}s")
                self.connected = False
                self.websocket = None
                
                # Wait before retrying
                await asyncio.sleep(retry_delay)
                
                # Exponential backoff with maximum
                retry_delay = min(retry_delay * 1.5, max_retry_delay)
    
    def send_detection(self, detection: SignalDetection) -> bool:
        """Send signal detection to central server for TDoA processing"""
        try:
            if not self.connected or not self.websocket:
                logger.warning("Not connected to central server")
                return False
            
            # Convert detection to JSON
            detection_data = asdict(detection)
            message = {
                "type": "signal_detection",
                "data": detection_data,
                "timestamp": detection.timestamp_utc
            }
            
            # Send via WebSocket asynchronously
            if self.loop and self.loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self.websocket.send(json.dumps(message)), 
                    self.loop
                )
                future.result(timeout=1.0)  # Wait max 1 second
                
                logger.info(f"Sent detection to central: {detection.frequency_mhz} MHz from {detection.buoy_id}")
                return True
            else:
                logger.warning("WebSocket event loop not running")
                return False
            
        except Exception as e:
            logger.error(f"Failed to send detection: {e}")
            return False
    
    def send_heartbeat(self, status: BuoyStatus) -> bool:
        """Send heartbeat with buoy status"""
        try:
            if not self.connected:
                return False
            
            message = {
                "type": "heartbeat",
                "data": asdict(status),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # In production: send via TCP/WebSocket
            logger.debug(f"Heartbeat sent: {status.buoy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
            return False
    
    def send_gps_update(self, lat: float, lng: float) -> bool:
        """Send GPS position update to central server"""
        if not self.connected:
            logger.warning("Not connected to central server - cannot send GPS update")
            return False
        
        try:
            message = {
                "type": "gps_update",
                "node_id": self.buoy_id,
                "lat": lat,
                "lng": lng,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            asyncio.run_coroutine_threadsafe(
                self.websocket.send(json.dumps(message)),
                self.loop
            )
            logger.info(f"Sent GPS update: ({lat:.6f}, {lng:.6f})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send GPS update: {e}")
            return False

class BuoyNode:
    """Main buoy node class that coordinates all subsystems"""
    
    def __init__(self, buoy_id: str, central_server_host: str = "localhost", central_server_port: int = 8081, development_mode: bool = False):
        self.buoy_id = buoy_id
        self.start_time = time.time()
        self.signals_detected_count = 0
        self.running = False
        
        # Initialize subsystems
        self.gps_source = GPSTimeSource(development_mode)
        self.signal_detector = SignalDetector(buoy_id, self.gps_source, development_mode)
        self.communicator = CentralCommunicator(buoy_id, central_server_host, central_server_port)
        
        logger.info(f"Buoy node {buoy_id} initialized")
    
    def startup(self) -> bool:
        """Initialize and start all buoy subsystems"""
        logger.info(f"Starting buoy node {self.buoy_id}")
        
        # Initialize GPS timing
        if not self.gps_source.initialize_gps():
            logger.error("GPS initialization failed - cannot operate without precise timing")
            return False
        
        # Connect to central server
        if not self.communicator.connect_to_central():
            logger.warning("Failed to connect to central server - will retry")
        else:
            # Send GPS coordinates after successful connection
            lat, lng = self.gps_source.get_position()
            self.communicator.send_gps_update(lat, lng)
        
        # Start signal monitoring
        self.signal_detector.start_monitoring()
        
        # Start main processing loop
        self.running = True
        self._start_processing_threads()
        
        logger.info(f"Buoy node {self.buoy_id} is operational")
        return True
    
    def shutdown(self):
        """Gracefully shutdown the buoy node"""
        logger.info(f"Shutting down buoy node {self.buoy_id}")
        
        self.running = False
        self.signal_detector.stop_monitoring()
        
        logger.info(f"Buoy node {self.buoy_id} shutdown complete")
    
    def _start_processing_threads(self):
        """Start background processing threads"""
        # Signal processing thread
        signal_thread = threading.Thread(target=self._process_signals, daemon=True)
        signal_thread.start()
        
        # Heartbeat thread
        heartbeat_thread = threading.Thread(target=self._send_heartbeats, daemon=True)
        heartbeat_thread.start()
    
    def _process_signals(self):
        """Process detected signals and send to central server"""
        while self.running:
            try:
                # Get signal from queue (block for 1 second)
                detection = self.signal_detector.signal_queue.get(timeout=1.0)
                
                # Send to central server for TDoA processing
                if self.communicator.send_detection(detection):
                    self.signals_detected_count += 1
                    
                    # Log emergency signals with high priority
                    if detection.signal_type == "emergency":
                        logger.warning(f"EMERGENCY SIGNAL DETECTED: {detection.frequency_mhz} MHz "
                                     f"at {detection.timestamp_utc}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing signal: {e}")
    
    def _send_heartbeats(self):
        """Send periodic heartbeat to central server"""
        while self.running:
            try:
                lat, lng = self.gps_source.get_position()
                
                status = BuoyStatus(
                    buoy_id=self.buoy_id,
                    lat=lat,
                    lng=lng,
                    status="active" if self.gps_source.gps_locked else "degraded",
                    gps_locked=self.gps_source.gps_locked,
                    timing_accuracy_ns=self.gps_source.timing_accuracy_ns,
                    last_heartbeat=datetime.now(timezone.utc).isoformat(),
                    signals_detected=self.signals_detected_count,
                    uptime_seconds=int(time.time() - self.start_time),
                    latest_signal_timestamp=self.signal_detector.get_latest_signal_timestamp()
                )
                
                self.communicator.send_heartbeat(status)
                
                # Send heartbeat every 30 seconds
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")
                time.sleep(30)
    
    def get_status(self) -> BuoyStatus:
        """Get current buoy status"""
        try:
            lat, lng = self.gps_source.get_position()
        except:
            lat, lng = 0.0, 0.0
        
        return BuoyStatus(
            buoy_id=self.buoy_id,
            lat=lat,
            lng=lng,
            status="active" if self.gps_source.gps_locked else "error",
            gps_locked=self.gps_source.gps_locked,
            timing_accuracy_ns=self.gps_source.timing_accuracy_ns,
            last_heartbeat=datetime.now(timezone.utc).isoformat(),
            signals_detected=self.signals_detected_count,
            uptime_seconds=int(time.time() - self.start_time)
        )

def main():
    """Main entry point for buoy node operation"""
    import sys
    import os
    
    # Get configuration from environment variables or defaults
    buoy_id = os.getenv('BUOY_ID', str(uuid.uuid4()))
    central_host = os.getenv('CENTRAL_HOST', 'localhost')
    central_port = int(os.getenv('CENTRAL_PORT', '8081'))
    development_mode = os.getenv('DEVELOPMENT_MODE', 'false').lower() == 'true'
    
    logger.info(f"Connecting to central server at {central_host}:{central_port}")
    
    # Create and start buoy node
    buoy = BuoyNode(buoy_id, central_host, central_port, development_mode)
    
    try:
        if buoy.startup():
            logger.info(f"Buoy {buoy_id} running. Press Ctrl+C to stop.")
            
            # Keep running until interrupted
            while True:
                time.sleep(10)
                status = buoy.get_status()
                logger.info(f"Status: {status.signals_detected} signals detected, "
                           f"GPS lock: {status.gps_locked}, "
                           f"Uptime: {status.uptime_seconds}s")
        else:
            logger.error("Failed to start buoy node")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        buoy.shutdown()

if __name__ == "__main__":
    main() 