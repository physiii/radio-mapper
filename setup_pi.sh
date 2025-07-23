#!/bin/bash

# Radio-Mapper Raspberry Pi Setup Script
# This script sets up the Radio-Mapper system on a Raspberry Pi

set -e

echo "🚀 Setting up Radio-Mapper on Raspberry Pi..."

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo; then
    echo "⚠️  Warning: This script is designed for Raspberry Pi"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo "✅ Docker installed. Please log out and back in for group changes to take effect."
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo "🐳 Installing Docker Compose..."
    sudo apt-get install -y docker-compose-plugin
fi

# Install required packages for RTL-SDR
echo "📡 Installing RTL-SDR dependencies..."
sudo apt-get install -y \
    libusb-1.0-0-dev \
    pkg-config \
    cmake \
    build-essential \
    git

# Create udev rules for RTL-SDR
echo "🔧 Setting up RTL-SDR udev rules..."
sudo tee /etc/udev/rules.d/20-rtlsdr.rules > /dev/null <<EOF
SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2832", GROUP="plugdev", MODE="0666"
EOF

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Create data directory
echo "📁 Creating data directory..."
mkdir -p data

# Set up system optimizations for Pi
echo "⚡ Optimizing system for radio operations..."

# Increase USB buffer sizes
echo 'net.core.rmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem = 4096 87380 16777216' | sudo tee -a /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem = 4096 65536 16777216' | sudo tee -a /etc/sysctl.conf

# Apply sysctl changes
sudo sysctl -p

# Set CPU governor to performance for better real-time performance
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils

# Create systemd service for auto-start
echo "🔧 Creating systemd service for auto-start..."
sudo tee /etc/systemd/system/radio-mapper.service > /dev/null <<EOF
[Unit]
Description=Radio-Mapper System
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/docker compose -f docker-compose.pi.yml up -d
ExecStop=/usr/bin/docker compose -f docker-compose.pi.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

# Enable the service
sudo systemctl enable radio-mapper.service

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Connect your RTL-SDR device via USB"
echo "2. Connect your GPS module (optional)"
echo "3. Edit config.yaml with your settings"
echo "4. Start the system:"
echo "   docker compose -f docker-compose.pi.yml up -d"
echo ""
echo "🌐 Access the web interface at: http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "📊 Monitor system status:"
echo "   docker compose -f docker-compose.pi.yml ps"
echo "   docker compose -f docker-compose.pi.yml logs -f"
echo ""
echo "🔄 Auto-start is enabled. The system will start automatically on boot." 