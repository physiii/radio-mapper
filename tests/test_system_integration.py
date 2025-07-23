#!/usr/bin/env python3
"""
Comprehensive Radio-Mapper System Integration Tests

This test suite verifies that all components of the Radio-Mapper system
are working correctly together.
"""

import asyncio
import json
import requests
import time
import sys
import os
import websockets
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class RadioMapperSystemTest:
    def __init__(self):
        self.central_host = "localhost"
        self.central_http_port = 5001
        self.central_ws_port = 8081
        self.webapp_port = 5000
        
        self.test_results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "failures": []
        }
    
    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def run_test(self, test_name, test_func):
        """Run a single test and track results"""
        self.test_results["total_tests"] += 1
        self.log(f"Running test: {test_name}")
        
        try:
            result = test_func()
            if result:
                self.test_results["passed"] += 1
                self.log(f"✅ PASSED: {test_name}")
                return True
            else:
                self.test_results["failed"] += 1
                self.test_results["failures"].append(test_name)
                self.log(f"❌ FAILED: {test_name}", "ERROR")
                return False
        except Exception as e:
            self.test_results["failed"] += 1
            self.test_results["failures"].append(f"{test_name}: {str(e)}")
            self.log(f"❌ FAILED: {test_name} - Exception: {str(e)}", "ERROR")
            return False
    
    def test_central_processor_http_alive(self):
        """Test if central processor HTTP server is responding"""
        try:
            response = requests.get(f"http://{self.central_host}:{self.central_http_port}/api/nodes", timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.log(f"Central processor HTTP test failed: {e}")
            return False
    
    def test_central_processor_websocket_alive(self):
        """Test if central processor WebSocket server is accepting connections"""
        async def check_websocket():
            try:
                uri = f"ws://{self.central_host}:{self.central_ws_port}"
                async with websockets.connect(uri, ping_interval=None) as websocket:
                    return True
            except Exception as e:
                self.log(f"WebSocket connection failed: {e}")
                return False
        
        try:
            return asyncio.run(check_websocket())
        except Exception as e:
            self.log(f"WebSocket test failed: {e}")
            return False
    
    def test_webapp_alive(self):
        """Test if webapp is responding"""
        try:
            response = requests.get(f"http://{self.central_host}:{self.webapp_port}/api/devices", timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.log(f"Webapp test failed: {e}")
            return False
    
    def test_webapp_connects_to_central(self):
        """Test if webapp can connect to central processor"""
        try:
            response = requests.get(f"http://{self.central_host}:{self.webapp_port}/api/devices", timeout=5)
            if response.status_code != 200:
                return False
            
            # If webapp returns mock data, it means it's NOT connected to central
            devices = response.json()
            
            # Check if we get back the specific mock device names
            if any(device.get('name') in ['OKC North', 'OKC East', 'OKC South'] for device in devices):
                self.log("Webapp is returning mock data - not connected to central processor")
                return False
            
            return True
        except Exception as e:
            self.log(f"Webapp-central connection test failed: {e}")
            return False
    
    def test_buoy_connection_simulation(self):
        """Test simulated buoy connection to central processor"""
        async def simulate_buoy():
            try:
                uri = f"ws://{self.central_host}:{self.central_ws_port}"
                async with websockets.connect(uri, ping_interval=None) as websocket:
                    # Send registration message
                    registration = {
                        "type": "registration",
                        "data": {
                            "node_id": "TEST_BUOY_1",
                            "lat": 35.4676,
                            "lng": -97.5164,
                            "capabilities": ["fm_scanner", "emergency_scanner"]
                        }
                    }
                    await websocket.send(json.dumps(registration))
                    
                    # Wait for response
                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    
                    # Send a test signal detection
                    detection = {
                        "type": "signal_detection",
                        "data": {
                            "node_id": "TEST_BUOY_1",
                            "frequency_mhz": 105.7,
                            "signal_strength_dbm": -45,
                            "lat": 35.4676,
                            "lng": -97.5164,
                            "gps_timestamp_ns": int(time.time() * 1e9),
                            "timestamp_utc": datetime.now().isoformat(),
                            "signal_type": "fm",
                            "confidence": 0.95
                        }
                    }
                    await websocket.send(json.dumps(detection))
                    
                    return True
                    
            except Exception as e:
                self.log(f"Buoy simulation failed: {e}")
                return False
        
        return asyncio.run(simulate_buoy())
    
    def test_central_processor_nodes_api(self):
        """Test central processor nodes API after simulated connection"""
        # First run the buoy simulation
        if not self.test_buoy_connection_simulation():
            return False
        
        # Wait a moment for registration to process
        time.sleep(2)
        
        try:
            response = requests.get(f"http://{self.central_host}:{self.central_http_port}/api/nodes", timeout=5)
            if response.status_code != 200:
                return False
            
            nodes = response.json()
            # Should have at least our test buoy
            return len(nodes) > 0 and any(node.get('id') == 'TEST_BUOY_1' for node in nodes)
            
        except Exception as e:
            self.log(f"Nodes API test failed: {e}")
            return False
    
    def test_hardware_detection(self):
        """Test if system can detect actual hardware"""
        try:
            # Import the config manager to test hardware detection
            from config_manager import ConfigManager
            config_manager = ConfigManager()
            interfaces = config_manager.auto_detect_interfaces()
            
            sdr_count = len(interfaces.get('sdr_devices', []))
            gps_count = len(interfaces.get('gps_devices', []))
            
            self.log(f"Detected {sdr_count} SDR devices, {gps_count} GPS devices")
            
            # For this test, we just verify the detection runs without error
            return True
            
        except Exception as e:
            self.log(f"Hardware detection test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests in sequence"""
        self.log("🚀 Starting Radio-Mapper System Integration Tests")
        self.log("=" * 60)
        
        # Basic connectivity tests
        self.run_test("Central Processor HTTP Server", self.test_central_processor_http_alive)
        self.run_test("Central Processor WebSocket Server", self.test_central_processor_websocket_alive)
        self.run_test("Webapp HTTP Server", self.test_webapp_alive)
        
        # Integration tests
        self.run_test("Webapp connects to Central Processor", self.test_webapp_connects_to_central)
        self.run_test("Central Processor Nodes API", self.test_central_processor_nodes_api)
        
        # Hardware tests
        self.run_test("Hardware Detection", self.test_hardware_detection)
        
        # Print results
        self.log("=" * 60)
        self.log("🏁 Test Results:")
        self.log(f"Total tests: {self.test_results['total_tests']}")
        self.log(f"Passed: {self.test_results['passed']}")
        self.log(f"Failed: {self.test_results['failed']}")
        
        if self.test_results['failures']:
            self.log("❌ Failed tests:")
            for failure in self.test_results['failures']:
                self.log(f"  - {failure}")
        
        success_rate = (self.test_results['passed'] / self.test_results['total_tests']) * 100
        self.log(f"Success rate: {success_rate:.1f}%")
        
        if self.test_results['failed'] == 0:
            self.log("🎉 ALL TESTS PASSED!")
            return True
        else:
            self.log("💥 SOME TESTS FAILED - System needs debugging")
            return False

if __name__ == "__main__":
    tester = RadioMapperSystemTest()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 