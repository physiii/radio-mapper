# Radio-Mapper Buoy Deployment Guide

## Overview

The Radio-Mapper Emergency Response System uses distributed "buoys" - radio monitoring stations that work together to triangulate emergency signals using Time Difference of Arrival (TDoA) techniques.

## Terminology

- **Buoy**: A radio monitoring station with RTL-SDR + GPS (like maritime buoys)
- **Central Server**: Processes data from all buoys and performs triangulation
- **TDoA**: Time Difference of Arrival - how we calculate signal location

## Oklahoma City Deployment Example

### Minimum Requirements for Triangulation
- **3+ buoys** minimum for 2D triangulation
- **4+ buoys** recommended for accuracy and redundancy
- **GPS timing** for sub-microsecond synchronization
- **50km maximum** separation between buoys

### Suggested Buoy Locations

```
     🟢 North Buoy (Edmond)
       35.5276, -97.5164

🟢 West        🏛️ OKC        🟢 East Buoy  
35.4676        Downtown      35.4676
-97.6164       35.4676       -97.4164
               -97.5164    

     🟢 South Buoy (Moore)
       35.3776, -97.5164
```

## Hardware Setup (Per Buoy)

### Required Hardware
- **Raspberry Pi 4** (or similar single-board computer)
- **RTL-SDR dongle** (RTL2832U based)
- **GPS module** with PPS output (for timing)
- **Antenna** suitable for emergency frequencies
- **Internet connection** (WiFi or Ethernet)

### Emergency Frequencies Monitored
- **121.5 MHz** - Aviation emergency
- **243.0 MHz** - Military emergency (ELT)
- **406.025 MHz** - EPIRB/PLB emergency beacons
- **156.8 MHz** - Marine emergency (Channel 16)
- **462.675 MHz** - GMRS emergency

## Software Deployment

### 1. Central Server (Main Machine)

```bash
# Clone and setup
git clone <repository>
cd radio-mapper
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure for server
cp config.yaml config-server.yaml
# Edit config-server.yaml:
#   - Set bind_host: "0.0.0.0" 
#   - Note your public IP for buoys

# Start central server
python3 run.py server --config config-server.yaml
```

Your server will be available at:
- **WebSocket**: `ws://YOUR-IP:8081` (for buoys)
- **Web Dashboard**: `http://YOUR-IP:5000` (for monitoring)
- **API**: `http://YOUR-IP:5001` (for data access)

### 2. Buoy Configuration (Each Raspberry Pi)

```bash
# On each Raspberry Pi
git clone <repository>
cd radio-mapper
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup hardware detection
python3 run.py setup

# Use appropriate config for location
cp config-buoy-north.yaml config.yaml    # For north buoy
# OR
cp config-buoy-south.yaml config.yaml    # For south buoy
# OR  
cp config-buoy-east.yaml config.yaml     # For east buoy

# Edit config.yaml to set your central server IP:
#   websocket_url: "ws://YOUR-SERVER-IP:8081"
#   http_url: "http://YOUR-SERVER-IP:5001"

# Start buoy client
python3 run.py client
```

### 3. Individual Buoy Commands

```bash
# North buoy
python3 run.py client --config config-buoy-north.yaml

# South buoy  
python3 run.py client --config config-buoy-south.yaml

# East buoy
python3 run.py client --config config-buoy-east.yaml
```

## System Operation

### Starting the Complete System

1. **Start Central Server** (main machine):
   ```bash
   python3 run.py server
   ```

2. **Start Each Buoy** (on Raspberry Pis):
   ```bash
   python3 run.py client
   ```

3. **Open Web Dashboard**:
   ```bash
   # On any machine
   python3 run.py web
   # Then browse to http://localhost:5000
   ```

### Monitoring Operations

The web dashboard shows:
- **Active Buoys**: Number of connected monitoring stations
- **Real-time Signals**: Emergency signals being detected
- **Triangulation**: Calculated positions of signal sources
- **System Health**: Timing accuracy and connection status

### Emergency Response

When emergency signals are detected:
1. **Automatic Detection**: System monitors emergency frequencies
2. **Multi-Buoy Correlation**: Signals detected by 3+ buoys are triangulated
3. **Position Calculation**: TDoA algorithms calculate transmitter location
4. **Real-time Alerts**: Web dashboard shows immediate notifications
5. **Confidence Scoring**: Each detection includes accuracy estimation

## Network Configuration

### Firewall Setup (Central Server)
```bash
# Allow buoy connections
sudo ufw allow 8081  # WebSocket
sudo ufw allow 5001  # HTTP API
sudo ufw allow 5000  # Web dashboard
```

### Port Forwarding (If Behind Router)
- **8081** - WebSocket (buoys connect here)
- **5001** - HTTP API (data access)
- **5000** - Web dashboard (monitoring)

## Time Synchronization

### GPS Timing (Recommended)
- **Accuracy**: Sub-microsecond
- **Requirement**: GPS module with PPS output
- **Configuration**: Set `timing.method: "gps"` in config

### NTP Fallback
- **Accuracy**: 1-10 milliseconds  
- **Use case**: When GPS unavailable
- **Configuration**: Set `timing.method: "ntp"` in config

## Troubleshooting

### Common Issues

**Buoy won't connect to server:**
- Check `websocket_url` in config matches server IP
- Verify firewall allows port 8081
- Ensure server is running and accessible

**No GPS signal:**
- Check GPS antenna placement (clear sky view)
- Verify GPS device path in config (`/dev/ttyUSB0`)
- Use `python3 run.py setup` to detect devices

**Poor triangulation accuracy:**
- Ensure 3+ buoys detecting same signal
- Check timing synchronization (GPS preferred)
- Verify buoy spacing (not too close together)

**No SDR device found:**
- Check RTL-SDR is plugged in
- Run `rtl_test -t` to verify device
- Try different USB port

### Testing Commands

```bash
# Test configuration
python3 run.py test

# Setup and detect hardware  
python3 run.py setup

# Check system status
tail -f radio-mapper.log
```

## Production Considerations

### Security
- Use VPN for buoy-to-server communications
- Implement authentication for web dashboard
- Regular security updates on all systems

### Reliability  
- Battery backup for power outages
- Automatic restart scripts
- Health monitoring and alerting
- Redundant internet connections

### Legal Compliance
- Verify emergency frequency monitoring is legal in your jurisdiction
- Coordinate with local emergency services
- Ensure proper licensing for radio equipment

This deployment guide provides the framework for a professional emergency response triangulation system using distributed buoy networks around Oklahoma City or any metropolitan area. 