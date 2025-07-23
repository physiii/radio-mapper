# Radio-Mapper for Raspberry Pi 🍓

This guide will help you deploy the Radio-Mapper system on a Raspberry Pi for emergency signal triangulation.

## 🎯 What You'll Get

- **Real-time signal detection** using RTL-SDR
- **GPS-synchronized TDoA triangulation** across multiple Pi nodes
- **Web interface** for monitoring and control
- **Automatic startup** on boot
- **Resource-optimized** for Pi performance

## 📋 Requirements

### Hardware
- **Raspberry Pi 4** (4GB RAM recommended, 8GB optimal)
- **RTL-SDR dongle** (RTL2832U chipset)
- **GPS module** (optional, for precise timing)
- **MicroSD card** (32GB+ recommended)
- **Power supply** (3A+ recommended for stable operation)

### Software
- **Raspberry Pi OS** (Bullseye or newer)
- **Docker** (will be installed by setup script)
- **Docker Compose** (will be installed by setup script)

## 🚀 Quick Start

### 1. Prepare Your Pi

```bash
# Update your Pi
sudo apt update && sudo apt upgrade -y

# Clone the repository
git clone https://github.com/your-repo/radio-mapper.git
cd radio-mapper
```

### 2. Run the Setup Script

```bash
# Run the automated setup
./setup_pi.sh
```

This script will:
- Install Docker and Docker Compose
- Set up RTL-SDR udev rules
- Configure system optimizations
- Create auto-start service
- Set up data directories

### 3. Configure Your System

Edit `config.yaml` with your settings:

```yaml
# Example configuration for Pi deployment
buoy:
  name: "PI_BUOY_1"
  location:
    latitude: 35.4676  # Your latitude
    longitude: -97.5164  # Your longitude

scanning:
  frequency_ranges:
    - start: 121.0
      end: 122.0
      type: "emergency"
    - start: 105.0
      end: 108.0
      type: "fm_radio"
  
  # Pi-optimized settings
  dwell_time_seconds: 2.0
  correlation_window_seconds: 5.0
```

### 4. Start the System

```bash
# Start all services
docker compose -f docker-compose.pi.yml up -d

# Check status
docker compose -f docker-compose.pi.yml ps

# View logs
docker compose -f docker-compose.pi.yml logs -f
```

### 5. Access the Web Interface

Open your browser and go to:
```
http://YOUR_PI_IP:5000
```

## 🔧 Pi-Specific Optimizations

### Resource Limits
The Pi compose file includes resource limits to prevent system overload:

- **Central Processor**: 512MB RAM, 0.5 CPU cores
- **Web Interface**: 256MB RAM, 0.25 CPU cores  
- **Buoy Nodes**: 256MB RAM, 0.5 CPU cores each

### Performance Tuning
- **CPU Governor**: Set to "performance" for real-time operations
- **Network Buffers**: Increased for better USB data throughput
- **ARM64 Optimization**: Uses ARM64 base image for better performance

### Auto-Start
The system automatically starts on boot via systemd service.

## 📡 Hardware Setup

### RTL-SDR Connection
1. Connect RTL-SDR to USB port
2. Verify detection: `lsusb | grep RTL`
3. Check permissions: `ls -l /dev/bus/usb/*/*`

### GPS Module (Optional)
1. Connect GPS to USB or serial port
2. Update `config.yaml` with correct device path
3. Test GPS: `python3 -c "import serial; print('GPS ready')"`

## 🎛️ Management Commands

### Start/Stop Services
```bash
# Start all services
docker compose -f docker-compose.pi.yml up -d

# Stop all services
docker compose -f docker-compose.pi.yml down

# Restart specific service
docker compose -f docker-compose.pi.yml restart buoy-1
```

### View Logs
```bash
# All services
docker compose -f docker-compose.pi.yml logs -f

# Specific service
docker compose -f docker-compose.pi.yml logs -f central-processor

# Recent logs
docker compose -f docker-compose.pi.yml logs --tail=50
```

### System Status
```bash
# Container status
docker compose -f docker-compose.pi.yml ps

# Resource usage
docker stats

# System resources
htop
```

## 🔍 Troubleshooting

### RTL-SDR Not Detected
```bash
# Check USB devices
lsusb

# Check udev rules
ls -l /etc/udev/rules.d/20-rtlsdr.rules

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### High CPU Usage
```bash
# Check resource limits
docker stats

# Reduce scanning frequency in config.yaml
# Increase dwell_time_seconds
```

### Memory Issues
```bash
# Check memory usage
free -h

# Reduce number of buoy containers
# Edit docker-compose.pi.yml to remove buoy-2, buoy-3
```

### Network Issues
```bash
# Check network connectivity
ping 8.8.8.8

# Check port availability
netstat -tlnp | grep :5000
```

## 📊 Monitoring

### Web Interface Features
- **Real-time map** of buoy locations
- **Signal detection** history
- **System status** monitoring
- **GPS lock** status
- **TDoA triangulation** results

### API Endpoints
- `GET /api/nodes` - Connected buoy nodes
- `GET /api/signals` - Detected signals
- `GET /api/system-status` - Overall system health

## 🔄 Updates

### Update the System
```bash
# Pull latest code
git pull

# Rebuild containers
docker compose -f docker-compose.pi.yml down
docker compose -f docker-compose.pi.yml build
docker compose -f docker-compose.pi.yml up -d
```

### Backup Data
```bash
# Backup configuration and data
tar -czf radio-mapper-backup-$(date +%Y%m%d).tar.gz \
    config.yaml data/ docker-compose.pi.yml
```

## 🆘 Emergency Procedures

### System Recovery
```bash
# Force restart all services
sudo systemctl restart radio-mapper.service

# Reset to factory settings
docker compose -f docker-compose.pi.yml down
docker system prune -a
./setup_pi.sh
```

### Emergency Signal Detection
The system automatically scans emergency frequencies:
- **121.5 MHz** - Aviation emergency
- **243.0 MHz** - Military emergency
- **156.8 MHz** - Maritime emergency

## 📞 Support

For issues specific to Pi deployment:
1. Check the troubleshooting section above
2. Review logs: `docker compose -f docker-compose.pi.yml logs`
3. Verify hardware connections
4. Check system resources: `htop`, `free -h`

## 🎯 Performance Tips

1. **Use Pi 4 with 4GB+ RAM** for optimal performance
2. **Connect RTL-SDR directly** to Pi (avoid USB hubs)
3. **Use quality power supply** (3A+ recommended)
4. **Keep Pi cool** - add heatsink/fan if needed
5. **Use Class 10+ microSD** for better I/O performance

---

**Ready to deploy emergency signal triangulation on your Raspberry Pi!** 🚀 