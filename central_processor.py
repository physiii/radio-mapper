#!/usr/bin/env python3
"""
Central Signal Processing Server for Radio-Mapper

This server:
1. Receives IQ detections from multiple distributed nodes
2. Correlates signals across nodes using timestamps and signatures
3. Performs real TDoA triangulation when 3+ nodes detect same signal
4. Serves web interface with live triangulation results
5. Handles user signal search requests
"""

import asyncio
import websockets
import json
import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
import numpy as np
from flask import Flask, jsonify, request, render_template_string
from tdoa_processor import TDoAProcessor, SignalDetection as TDoADetection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NodeConnection:
    """Information about a connected node"""
    node_id: str
    websocket: websockets.WebSocketServerProtocol
    last_seen: datetime
    position: tuple  # (lat, lng)
    status: str = "active"
    latest_signal_timestamp: Optional[datetime] = None

@dataclass
class LiveSignalDetection:
    """Real-time signal detection from a node"""
    node_id: str
    frequency_mhz: float
    signal_strength_dbm: float
    timestamp_utc: str
    gps_timestamp_ns: int
    lat: float
    lng: float
    confidence: float
    signal_type: str
    bandwidth_hz: float = 10000
    iq_samples: Optional[List[complex]] = None

@dataclass
class TriangulatedSignal:
    """Signal triangulated from multiple nodes"""
    signal_id: str
    frequency_mhz: float
    estimated_lat: float
    estimated_lng: float
    confidence: float
    detected_by: List[str]
    detection_timestamps: List[str]
    signal_type: str
    triangulation_method: str
    accuracy_meters: float

class CentralProcessor:
    """Central processing server that coordinates multiple nodes"""
    
    def __init__(self, host: str = "localhost", ws_port: int = 8080, http_port: int = 5001):
        self.host = host
        self.ws_port = ws_port
        self.http_port = http_port
        
        # Connected nodes
        self.nodes: Dict[str, NodeConnection] = {}
        
        # Signal processing
        self.tdoa_processor = TDoAProcessor()
        self.signal_buffer: List[LiveSignalDetection] = []
        self.triangulated_signals: List[TriangulatedSignal] = []
        self.correlation_window_seconds = 5.0  # Signals within 5 seconds are correlated
        
        # Buffer management
        self.buffer_max_age_seconds = 24 * 60 * 60  # 24 hours
        self.buffer_cleanup_interval_seconds = 5 * 60  # 5 minutes
        
        # Flask app for web interface
        self.flask_app = Flask(__name__)
        self.setup_flask_routes()
        
        # Processing control
        self.running = False
        
    def setup_flask_routes(self):
        """Setup Flask routes for web API"""
        
        @self.flask_app.route('/api/nodes')
        def get_nodes():
            """Get list of connected nodes"""
            node_list = []
            for node_id, node in self.nodes.items():
                # Get the most recent coordinates from signal detections for this node
                latest_lat, latest_lng = node.position[0], node.position[1]
                
                # Look for recent signals from this node to get updated coordinates
                for detection in reversed(self.signal_buffer[-50:]):  # Check last 50 signals
                    if detection.node_id == node_id:
                        latest_lat = detection.lat
                        latest_lng = detection.lng
                        break  # Use the most recent coordinates
                
                node_list.append({
                    'id': node_id,
                    'name': node_id,
                    'lat': latest_lat,
                    'lng': latest_lng,
                    'status': node.status,
                    'lastSeen': node.last_seen.isoformat(),
                    'latest_signal_timestamp': node.latest_signal_timestamp.isoformat() if node.latest_signal_timestamp else None
                })
            return jsonify(node_list)
        
        @self.flask_app.route('/api/signals')
        def get_signals():
            """Get list of triangulated signals"""
            signal_list = []
            for signal in self.triangulated_signals[-50:]:  # Last 50 signals
                signal_list.append({
                    'id': signal.signal_id,
                    'frequency': signal.frequency_mhz,
                    'signal_strength': -50,  # Estimated
                    'lat': signal.estimated_lat,
                    'lng': signal.estimated_lng,
                    'detected_by': signal.detected_by,
                    'timestamp': signal.detection_timestamps[0],
                    'signal_type': signal.signal_type,
                    'classification': self._get_signal_classification(signal.frequency_mhz, signal.signal_type),
                    'confidence': signal.confidence,
                    'triangulated': True,
                    'accuracy_meters': signal.accuracy_meters
                })
            return jsonify(signal_list)
        
        @self.flask_app.route('/api/detections')
        def get_detections():
            """Get list of recent signal detections from all buoys"""
            detection_list = []
            
            # Get detections from last 10 minutes, grouped by frequency to ensure all frequencies are represented
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=10)
            recent_detections = []
            
            # Group by frequency to ensure balanced representation
            freq_groups = {}
            
            for detection in reversed(self.signal_buffer):  # Start from newest
                try:
                    detection_time = datetime.fromisoformat(detection.timestamp_utc.replace('Z', '+00:00'))
                    if detection_time < cutoff_time:
                        continue
                except (ValueError, TypeError):
                    # If timestamp is invalid, skip it
                    logger.warning(f"Skipping detection with invalid timestamp: {detection.timestamp_utc}")
                    continue

                freq = detection.frequency_mhz
                if freq not in freq_groups:
                    freq_groups[freq] = []
                    
                # Keep up to 20 most recent detections per frequency
                if len(freq_groups[freq]) < 20:
                    freq_groups[freq].append(detection)
            
            # Flatten the groups
            for freq_detections in freq_groups.values():
                recent_detections.extend(freq_detections)
            
            # Sort by timestamp (newest first)
            recent_detections.sort(key=lambda d: d.timestamp_utc, reverse=True)
            
            for i, detection in enumerate(recent_detections):
                detection_list.append({
                    'id': f"DET_{i}",
                    'frequency_mhz': detection.frequency_mhz,
                    'signal_strength_dbm': detection.signal_strength_dbm,
                    'lat': detection.lat,
                    'lng': detection.lng,
                    'node_id': detection.node_id,
                    'timestamp': detection.timestamp_utc,
                    'signal_type': detection.signal_type,
                    'confidence': detection.confidence,
                    'triangulated': False
                })
            return jsonify(detection_list)
        
        @self.flask_app.route('/api/search_signal', methods=['POST'])
        def search_signal():
            """Search for specific signal in historical data"""
            data = request.get_json()
            frequency = data.get('frequency_mhz')
            max_age_minutes = data.get('max_age_minutes', 60)
            
            # Search triangulated signals
            cutoff_time = datetime.now(timezone.utc).timestamp() - (max_age_minutes * 60)
            matches = []
            
            for signal in self.triangulated_signals:
                # Parse timestamp
                signal_time = datetime.fromisoformat(signal.detection_timestamps[0].replace('Z', '+00:00')).timestamp()
                
                if signal_time < cutoff_time:
                    continue
                
                # Check frequency match (±10 kHz tolerance)
                if abs(signal.frequency_mhz - frequency) < 0.01:
                    matches.append({
                        'frequency_mhz': signal.frequency_mhz,
                        'lat': signal.estimated_lat,
                        'lng': signal.estimated_lng,
                        'confidence': signal.confidence,
                        'detected_by': signal.detected_by,
                        'timestamp': signal.detection_timestamps[0],
                        'accuracy_meters': signal.accuracy_meters
                    })
            
            return jsonify({'matches': matches, 'count': len(matches)})
    
    def _get_signal_classification(self, frequency_mhz: float, signal_type: str) -> str:
        """Get human-readable signal classification"""
        if signal_type == "emergency":
            if abs(frequency_mhz - 121.5) < 0.001:
                return "Aviation Emergency - 121.5 MHz"
            elif abs(frequency_mhz - 243.0) < 0.001:
                return "Military Emergency - 243.0 MHz"
            else:
                return "Emergency Frequency"
        elif signal_type == "public_safety":
            return "Public Safety Radio"
        elif signal_type == "aviation":
            return "Aviation Communication"
        elif signal_type == "amateur":
            return "Amateur Radio"
        elif signal_type == "fm_radio":
            return "FM Radio Broadcast"
        else:
            return f"{signal_type.title()} Signal"
    
    async def handle_node_connection(self, websocket):
        """Handle WebSocket connection from a node"""
        node_id = None
        try:
            logger.info(f"New node connection from {websocket.remote_address}")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get('type')
                    
                    if msg_type == 'node_registration':
                        # Register new node
                        node_id = data['node_id']
                        position = (data.get('lat', 35.5513177334763), data.get('lng', -97.53220535352492))
                        
                        self.nodes[node_id] = NodeConnection(
                            node_id=node_id,
                            websocket=websocket,
                            last_seen=datetime.now(timezone.utc),
                            position=position
                        )
                        
                        logger.info(f"Node {node_id} registered at {position}")
                        
                        # Send acknowledgment
                        await websocket.send(json.dumps({
                            'type': 'registration_ack',
                            'status': 'registered',
                            'server_time': datetime.now(timezone.utc).isoformat()
                        }))
                    
                    elif msg_type == 'gps_update':
                        # Update node GPS position
                        node_id = data.get('node_id')
                        lat = data.get('lat')
                        lng = data.get('lng')
                        
                        if node_id and lat is not None and lng is not None:
                            if node_id in self.nodes:
                                old_position = self.nodes[node_id].position
                                self.nodes[node_id].position = (lat, lng)
                                logger.info(f"Updated GPS position for {node_id}: ({lat:.6f}, {lng:.6f}) (was {old_position[0]:.6f}, {old_position[1]:.6f})")
                            else:
                                logger.warning(f"Received GPS update for unknown node: {node_id}")
                        else:
                            logger.warning(f"Invalid GPS update message: {data}")
                    
                    elif msg_type == 'signal_detection':
                        # Process signal detection
                        detection_data = data['data']
                        
                        # Convert buoy_id to node_id for compatibility
                        if 'buoy_id' in detection_data:
                            detection_data['node_id'] = detection_data.pop('buoy_id')
                        
                        # Add default bandwidth if missing
                        if 'bandwidth_hz' not in detection_data:
                            detection_data['bandwidth_hz'] = 10000
                        
                        # Remove fields that LiveSignalDetection doesn't expect
                        unwanted_fields = ['iq_sample_file', 'correlation_id']
                        for field in unwanted_fields:
                            detection_data.pop(field, None)
                        
                        detection = LiveSignalDetection(**detection_data)
                        
                        # Update node last seen and latest signal timestamp
                        if detection.node_id in self.nodes:
                            self.nodes[detection.node_id].last_seen = datetime.now(timezone.utc)
                            self.nodes[detection.node_id].latest_signal_timestamp = datetime.fromisoformat(detection.timestamp_utc.replace('Z', '+00:00'))

                        # Add to signal buffer
                        self.signal_buffer.append(detection)
                        
                        logger.info(f"Signal from {detection.node_id}: {detection.frequency_mhz} MHz, {detection.signal_strength_dbm} dBm")
                        
                        # Trigger correlation processing
                        await self.process_signal_correlations()
                    
                    elif msg_type == 'heartbeat':
                        # Update last seen time
                        if node_id and node_id in self.nodes:
                            self.nodes[node_id].last_seen = datetime.now(timezone.utc)
                        
                        # Send heartbeat response
                        await websocket.send(json.dumps({
                            'type': 'heartbeat_ack',
                            'server_time': datetime.now(timezone.utc).isoformat()
                        }))
                
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from node: {message}")
                except Exception as e:
                    logger.error(f"Error processing message from node: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Node {node_id} disconnected")
        except Exception as e:
            logger.error(f"Error in node connection: {e}")
        finally:
            # Clean up disconnected node
            if node_id and node_id in self.nodes:
                del self.nodes[node_id]
                logger.info(f"Removed disconnected node {node_id}")
    
    async def process_signal_correlations(self):
        """Process signal buffer to find correlations and triangulate"""
        # This function is for real-time triangulation and should not modify the main buffer.
        now_ts = datetime.now(timezone.utc).timestamp()
        
        # Get a snapshot of recent signals for correlation
        correlation_candidates = []
        for det in reversed(self.signal_buffer):
            try:
                det_ts = datetime.fromisoformat(det.timestamp_utc.replace('Z', '+00:00')).timestamp()
                if now_ts - det_ts > self.correlation_window_seconds:
                    # Since the buffer is sorted by time, we can stop early.
                    break
                correlation_candidates.append(det)
            except (ValueError, TypeError):
                continue
        
        if len(correlation_candidates) < 3:
            return

        # Group signals by frequency and time window
        frequency_groups = {}
        for detection in correlation_candidates:
            # Group by frequency (±10 kHz tolerance)
            freq_key = round(detection.frequency_mhz, 2)
            if freq_key not in frequency_groups:
                frequency_groups[freq_key] = []
            frequency_groups[freq_key].append(detection)
        
        # Process each frequency group for triangulation
        for freq_mhz, detections in frequency_groups.items():
            if len(detections) >= 3:
                node_ids = set(d.node_id for d in detections)
                if len(node_ids) >= 3:
                    await self.triangulate_signal(detections)

    async def triangulate_signal(self, detections: List[LiveSignalDetection]):
        """Triangulate signal from multiple node detections"""
        try:
            # Convert to TDoA format
            tdoa_detections = []
            for detection in detections:
                tdoa_detection = TDoADetection(
                    buoy_id=detection.node_id,
                    frequency_mhz=detection.frequency_mhz,
                    gps_timestamp_ns=detection.gps_timestamp_ns,
                    lat=detection.lat,
                    lng=detection.lng,
                    signal_strength_dbm=detection.signal_strength_dbm,
                    timestamp_utc=detection.timestamp_utc,
                    confidence=detection.confidence
                )
                tdoa_detections.append(tdoa_detection)
            
            # Perform triangulation
            result = self.tdoa_processor.triangulate_signal(tdoa_detections)
            
            if result and result.estimated_lat and result.estimated_lng:
                # Create triangulated signal
                signal_id = f"SIG_{int(time.time())}_{len(self.triangulated_signals)}"
                
                triangulated = TriangulatedSignal(
                    signal_id=signal_id,
                    frequency_mhz=detections[0].frequency_mhz,
                    estimated_lat=result.estimated_lat,
                    estimated_lng=result.estimated_lng,
                    confidence=result.confidence,
                    detected_by=[d.node_id for d in detections],
                    detection_timestamps=[d.timestamp_utc for d in detections],
                    signal_type=detections[0].signal_type,
                    triangulation_method="TDoA",
                    accuracy_meters=result.accuracy_estimate_meters
                )
                
                self.triangulated_signals.append(triangulated)
                
                logger.warning(f"TRIANGULATED: {triangulated.frequency_mhz} MHz at "
                             f"({triangulated.estimated_lat:.6f}, {triangulated.estimated_lng:.6f}) "
                             f"± {triangulated.accuracy_meters:.0f}m")
                
                # Notify all connected nodes
                notification = {
                    'type': 'triangulation_result',
                    'data': asdict(triangulated)
                }
                
                for node in self.nodes.values():
                    try:
                        await node.websocket.send(json.dumps(notification))
                    except:
                        pass  # Ignore disconnected nodes
            
        except Exception as e:
            logger.error(f"Error in triangulation: {e}")
            
    def _periodic_buffer_cleanup(self):
        """Periodically clean up old signals from the buffer."""
        while self.running:
            time.sleep(self.buffer_cleanup_interval_seconds)
            
            if not self.signal_buffer:
                continue

            cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=self.buffer_max_age_seconds)
            
            initial_size = len(self.signal_buffer)
            
            # Since new signals are appended, we can find the first valid index
            first_valid_index = -1
            for i, detection in enumerate(self.signal_buffer):
                try:
                    detection_time = datetime.fromisoformat(detection.timestamp_utc.replace('Z', '+00:00'))
                    if detection_time >= cutoff_time:
                        first_valid_index = i
                        break
                except (ValueError, TypeError):
                    # Invalid format, keep for now, might be recent with bad format
                    continue

            if first_valid_index > 0:
                self.signal_buffer = self.signal_buffer[first_valid_index:]
                final_size = len(self.signal_buffer)
                logger.info(f"Cleaned up {initial_size - final_size} old signals from buffer.")
            elif first_valid_index == -1 and len(self.signal_buffer) > 0:
                # All signals are old
                self.signal_buffer = []
                logger.info(f"Cleaned up all {initial_size} old signals from buffer.")


    def start_flask_server(self):
        """Start Flask web server in background thread"""
        def run_flask():
            self.flask_app.run(host=self.host, port=self.http_port, debug=False)
        
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        logger.info(f"Flask server started on http://{self.host}:{self.http_port}")
    
    async def start_websocket_server(self):
        """Start WebSocket server for node connections"""
        logger.info(f"Starting WebSocket server on ws://{self.host}:{self.ws_port}")
        
        async with websockets.serve(
            self.handle_node_connection, 
            self.host, 
            self.ws_port,
            ping_interval=30,
            ping_timeout=10
        ):
            logger.info("Central processor server is operational")
            # Keep server running
            await asyncio.Future()  # Run forever
    
    def start(self):
        """Start the central processing server"""
        logger.info("Starting Central Signal Processing Server")
        
        # Start Flask web server
        self.start_flask_server()
        
        # Start WebSocket server
        self.running = True
        
        # Start periodic cleanup task
        cleanup_thread = threading.Thread(target=self._periodic_buffer_cleanup, daemon=True)
        cleanup_thread.start()
        
        try:
            asyncio.run(self.start_websocket_server())
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            self.running = False

def main():
    """Main entry point"""
    import sys
    import os
    
    # Load config to get defaults
    try:
        from config_manager import ConfigManager
        config = ConfigManager()
        server_config = config.get_server_config()
        default_host = server_config.bind_host
        default_ws_port = server_config.websocket_port
        default_http_port = server_config.http_port
    except:
        # Fallback if config loading fails
        default_host = "0.0.0.0"
        default_ws_port = 8081
        default_http_port = 4000
    
    # Priority: Environment variables > Config file > Command line args
    host = os.getenv('BIND_HOST', default_host)
    ws_port = int(os.getenv('WEBSOCKET_PORT', default_ws_port))
    http_port = int(os.getenv('HTTP_PORT', default_http_port))
    
    # Allow command line override (for backward compatibility)
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        ws_port = int(sys.argv[2])
    if len(sys.argv) > 3:
        http_port = int(sys.argv[3])
    
    logger.info(f"Starting central processor: {host}:{ws_port} (WS), {host}:{http_port} (HTTP)")
    
    # Create and start central processor
    processor = CentralProcessor(host, ws_port, http_port)
    
    try:
        processor.start()
    except KeyboardInterrupt:
        logger.info("Central processor stopped")

if __name__ == "__main__":
    main() 