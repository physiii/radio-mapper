# Radio-Mapper: Emergency Signal Detection & Triangulation System

A GPS-synchronized radio frequency monitoring system designed for emergency response and signal triangulation using Software Defined Radio (SDR) technology.

## 🎯 **System Status: Partially Operational**

- ✅ **Signal Detection**: Emergency & commercial frequencies are successfully detected by a single buoy.
- ❌ **Triangulation**: **NOT OPERATIONAL**. Requires a minimum of 3 buoys and critical algorithm fixes.
- ❌ **GPS Timing**: Uses system time, not true GPS PPS time. Insufficient for triangulation.

---

## 🚀 **Getting Started**

This guide provides instructions for setting up and running the Radio-Mapper project using either Docker (recommended) or a local Python environment.

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/radio-mapper.git
cd radio-mapper
```

### **2. Docker Deployment (Recommended)**

#### **Standard Deployment**
```bash
# Build and start the containers in detached mode
docker-compose up -d --build
```

#### **Raspberry Pi Deployment**
```bash
# Build and start the containers for the Pi
docker-compose -f docker-compose.pi.yml up -d --build
```

### **3. Local Python Deployment**

#### **Prerequisites**
- Python 3.8+
- `librtlsdr-dev` package

#### **Installation**
```bash
# 1. Install system dependencies (for Debian/Ubuntu)
sudo apt-get update
sudo apt-get install -y librtlsdr-dev

# 2. Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Python packages
pip install -r requirements.txt
```

#### **Running the System**
```bash
# Run the main application
python3 run.py
```

---

## 🌐 **Accessing the System**

-   **Web Interface**: [http://localhost:5000](http://localhost:5000)
-   **API Endpoint**: [http://localhost:5001](http://localhost:5001)

---

## 🧪 **Validation**

After starting the system, you can run the comprehensive validation suite to check its status.

```bash
# Navigate to the tests directory
cd tests

# Run the validation script
python3 test_comprehensive_system_validation.py
```
**Expected Outcome:** 4 out of 5 tests should pass. The triangulation test is expected to fail as the system requires a multi-buoy setup to be fully operational.

---

## 📂 **Project Documentation**

For more detailed information about the system's architecture, TDoA analysis, and development roadmap, please refer to the documents in the `/Documents` folder.
- **`TDOA_README.md`**: A deep dive into the TDoA system's current state and challenges.
- **`ROADMAP.md`**: The project's development plan and future goals.

