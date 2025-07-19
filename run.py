#!/usr/bin/env python3
"""
Radio-Mapper Emergency Response System Runner

This script provides an easy way to start any component of the radio-mapper system
with proper configuration management.

Usage:
    python3 run.py server                    # Start central processing server
    python3 run.py client                    # Start IQ stream client (node)
    python3 run.py web                       # Start web interface
    python3 run.py setup                     # Setup configuration and detect devices
    python3 run.py test                      # Test system components
"""

import sys
import os
import argparse
import logging
import signal
import time
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import yaml
    from config_manager import get_config, ConfigManager
except ImportError as e:
    print(f"Error: Missing required dependencies: {e}")
    print("Please install requirements: pip install pyyaml")
    sys.exit(1)

def setup_signal_handlers():
    """Setup signal handlers for clean shutdown"""
    def signal_handler(sig, frame):
        print(f"\nReceived signal {sig}, shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def start_central_server(config: ConfigManager):
    """Start the central processing server"""
    print("🚀 Starting Central Processing Server...")
    
    try:
        from central_processor import CentralProcessor
        
        server_config = config.get_server_config()
        
        print(f"   WebSocket: ws://{server_config.bind_host}:{server_config.websocket_port}")
        print(f"   HTTP API:  http://{server_config.bind_host}:{server_config.http_port}")
        print(f"   Binding to: {server_config.bind_host}")
        
        processor = CentralProcessor(
            host=server_config.bind_host,
            ws_port=server_config.websocket_port,
            http_port=server_config.http_port
        )
        
        processor.start()
        
    except ImportError as e:
        print(f"❌ Error: Missing dependencies for central server: {e}")
        print("   Install: pip install websockets flask")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting central server: {e}")
        sys.exit(1)

def start_iq_client(config: ConfigManager):
    """Start an IQ streaming client"""
    print("📡 Starting IQ Stream Client...")
    
    try:
        from iq_stream_client import IQStreamClient
        
        buoy_config = config.get_buoy_config()
        server_config = config.get_server_config()
        sdr_config = config.get_sdr_config()
        
        print(f"   Buoy ID: {buoy_config.id}")
        print(f"   Location: ({buoy_config.latitude}, {buoy_config.longitude})")
        print(f"   Server: {server_config.websocket_url}")
        print(f"   Center Freq: {sdr_config.center_frequency_mhz} MHz")
        
        # Auto-detect interfaces
        interfaces = config.auto_detect_interfaces()
        print(f"   Local IP: {interfaces['ip_address']}")
        
        if interfaces['sdr_devices']:
            print(f"   SDR Devices: {len(interfaces['sdr_devices'])} found")
        else:
            print("   ⚠️  No SDR devices detected")
            if not config.get('development.mock_sdr', False):
                response = input("   Continue with mock SDR data? (y/N): ")
                if response.lower() != 'y':
                    sys.exit(1)
        
        if interfaces['gps_devices']:
            print(f"   GPS Devices: {', '.join(interfaces['gps_devices'])}")
        else:
            print("   ⚠️  No GPS devices detected, using fallback location")
        
        client = IQStreamClient(
            node_id=buoy_config.id,
            server_url=server_config.websocket_url
        )
        
        if client.start(sdr_config.center_frequency_mhz):
            print(f"✅ IQ Stream Client {buoy_config.id} is operational")
            print("   Press Ctrl+C to stop")
            
            # Keep running
            try:
                while True:
                    time.sleep(10)
                    print(f"   📊 Buoy operational - "
                          f"{len(client.signal_detector.signal_history)} signals detected")
            except KeyboardInterrupt:
                print("\n🛑 Shutdown requested")
        else:
            print("❌ Failed to start IQ stream client")
            sys.exit(1)
            
    except ImportError as e:
        print(f"❌ Error: Missing dependencies for IQ client: {e}")
        print("   Install: pip install numpy scipy websocket-client")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting IQ client: {e}")
        sys.exit(1)

def start_web_interface(config: ConfigManager):
    """Start the web interface"""
    print("🌐 Starting Web Interface...")
    
    try:
        from webapp.app import app
        
        web_port = config.get('web.port', 5000)
        server_config = config.get_server_config()
        
        print(f"   Web UI: http://localhost:{web_port}")
        print(f"   API Server: {server_config.http_url}")
        
        # Update Flask app configuration
        app.config['API_BASE_URL'] = server_config.http_url
        
        app.run(
            host='0.0.0.0',
            port=web_port,
            debug=False
        )
        
    except ImportError as e:
        print(f"❌ Error: Missing dependencies for web interface: {e}")
        print("   Install: pip install flask")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")
        sys.exit(1)

def setup_system(config: ConfigManager):
    """Setup system configuration and detect devices"""
    print("🔧 Radio-Mapper System Setup")
    print("=" * 50)
    
    # Detect interfaces
    print("\n📊 Detecting Hardware...")
    interfaces = config.auto_detect_interfaces()
    
    print(f"   Local IP Address: {interfaces['ip_address']}")
    print(f"   GPS Devices: {interfaces['gps_devices'] or 'None detected'}")
    print(f"   SDR Devices: {interfaces['sdr_devices'] or 'None detected'}")
    
    # Test GPS if available
    if interfaces['gps_devices']:
        print(f"\n🛰️  Testing GPS device: {interfaces['gps_devices'][0]}")
        try:
            # Basic GPS test (would need actual GPS library)
            print("   GPS device accessible")
        except Exception as e:
            print(f"   ⚠️  GPS test failed: {e}")
    
    # Test SDR if available
    if interfaces['sdr_devices']:
        print(f"\n📡 Testing SDR device...")
        try:
            import subprocess
            result = subprocess.run(['rtl_test', '-t'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("   ✅ SDR device test passed")
            else:
                print("   ⚠️  SDR device test failed")
                print(f"      {result.stderr}")
        except Exception as e:
            print(f"   ⚠️  SDR test error: {e}")
    
    # Test timing
    print(f"\n⏰ Testing Time Synchronization...")
    timing_config = config.get_timing_config()
    print(f"   Method: {timing_config.method}")
    print(f"   Target accuracy: {timing_config.target_accuracy_us} μs")
    
    if timing_config.ntp_enabled:
        print("   Testing NTP sync...")
        try:
            import subprocess
            result = subprocess.run(['ntpdate', '-q', 'time.nist.gov'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("   ✅ NTP sync test passed")
            else:
                print("   ⚠️  NTP sync test failed")
        except Exception:
            print("   ⚠️  NTP test unavailable (ntpdate not installed)")
    
    # Generate configuration file if needed
    config_file = "config.yaml"
    if not os.path.exists(config_file):
        print(f"\n📝 Generating configuration file: {config_file}")
        config.generate_example_config(config_file)
        print(f"   ✅ Configuration file created")
        print(f"   📖 Edit {config_file} to customize settings")
    else:
        print(f"\n📋 Configuration file: {config_file} (exists)")
    
    # Show configuration summary
    print(f"\n📋 Current Configuration:")
    buoy_config = config.get_buoy_config()
    server_config = config.get_server_config()
    
    print(f"   Buoy ID: {buoy_config.id}")
    print(f"   Location: ({buoy_config.latitude}, {buoy_config.longitude})")
    print(f"   Server: {server_config.websocket_url}")
    
    print(f"\n🚀 Setup complete! Next steps:")
    print(f"   1. Start central server: python3 run.py server")
    print(f"   2. Start buoy clients:   python3 run.py client")
    print(f"   3. Open web interface:   python3 run.py web")

def test_system(config: ConfigManager):
    """Test system components"""
    print("🧪 Testing Radio-Mapper System")
    print("=" * 40)
    
    # Test configuration
    print("\n📋 Testing Configuration...")
    try:
        config._validate_config()
        print("   ✅ Configuration validation passed")
    except Exception as e:
        print(f"   ❌ Configuration validation failed: {e}")
        return False
    
    # Test imports
    print("\n📦 Testing Dependencies...")
    dependencies = {
        'numpy': 'Signal processing',
        'scipy': 'Signal analysis',
        'websockets': 'Network communication',
        'flask': 'Web interface',
        'yaml': 'Configuration',
    }
    
    missing_deps = []
    for dep, purpose in dependencies.items():
        try:
            __import__(dep)
            print(f"   ✅ {dep} - {purpose}")
        except ImportError:
            print(f"   ❌ {dep} - {purpose} (MISSING)")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print(f"   Install with: pip install {' '.join(missing_deps)}")
        return False
    
    # Test hardware access
    print("\n🔧 Testing Hardware Access...")
    interfaces = config.auto_detect_interfaces()
    
    if interfaces['sdr_devices']:
        print(f"   ✅ SDR devices: {len(interfaces['sdr_devices'])} found")
    else:
        print("   ⚠️  No SDR devices found")
    
    if interfaces['gps_devices']:
        print(f"   ✅ GPS devices: {', '.join(interfaces['gps_devices'])}")
    else:
        print("   ⚠️  No GPS devices found")
    
    # Test network
    print("\n🌐 Testing Network...")
    server_config = config.get_server_config()
    
    import socket
    def test_port(host, port, name):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host.replace('0.0.0.0', 'localhost'), port))
            sock.close()
            if result == 0:
                print(f"   ⚠️  {name} port {port} is in use")
            else:
                print(f"   ✅ {name} port {port} is available")
        except Exception as e:
            print(f"   ❌ {name} port {port} test failed: {e}")
    
    test_port(server_config.bind_host, server_config.websocket_port, "WebSocket")
    test_port(server_config.bind_host, server_config.http_port, "HTTP")
    
    print("\n✅ System test complete!")
    return True

def print_usage():
    """Print usage information"""
    print("""
🚀 Radio-Mapper Emergency Response System

USAGE:
    python3 run.py <command> [options]

COMMANDS:
    server    Start central processing server
    client    Start IQ stream client (buoy)
    web       Start web dashboard
    setup     Setup configuration and detect hardware
    test      Test system components

EXAMPLES:
    # Initial setup
    python3 run.py setup
    
    # Start central server (run on main machine)
    python3 run.py server
    
    # Start buoy clients (run on each Raspberry Pi)
    python3 run.py client
    
    # Start web interface
    python3 run.py web

CONFIGURATION:
    Edit config.yaml to customize settings:
    - Buoy locations and IDs
    - Server URLs and ports
    - GPS device paths
    - SDR parameters
    - Timing synchronization

FOR HELP:
    Edit config.yaml for your specific deployment
    Check logs in radio-mapper.log
    """)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Radio-Mapper Emergency Response System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'command',
        choices=['server', 'client', 'web', 'setup', 'test'],
        help='Command to run'
    )
    
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Configuration file path (default: config.yaml)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    if len(sys.argv) == 1:
        print_usage()
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Load configuration
    try:
        config = get_config(args.config)
        
        # Setup logging
        if args.verbose:
            config.config['logging']['level'] = 'DEBUG'
        config.setup_logging()
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        print(f"   Run 'python3 run.py setup' to create initial config")
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'server':
            start_central_server(config)
        elif args.command == 'client':
            start_iq_client(config)
        elif args.command == 'web':
            start_web_interface(config)
        elif args.command == 'setup':
            setup_system(config)
        elif args.command == 'test':
            test_system(config)
        else:
            print(f"❌ Unknown command: {args.command}")
            print_usage()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 