# Radio-Mapper Test Suite

This directory contains comprehensive tests for the Radio-Mapper system.

## 🧪 **Core Test Files**

### **System Validation**
- **`test_comprehensive_system_validation.py`** - Complete end-to-end system validation
  - Validates frequency cycling, search functionality, signal quality
  - Explains expected behavior and system readiness
  - **Run this first** to validate overall system health

### **Integration Tests**
- **`test_system_integration.py`** - Core system integration tests
  - Buoy-to-central processor communication
  - Basic signal detection and storage
  - WebSocket connectivity

### **Hardware Tests**
- **`test_hardware_detection.py`** - Hardware compatibility tests
  - SDR device detection
  - GPS device detection
  - System requirements validation

### **TDoA Analysis**
- **`test_tdoa_validation.py`** - Triangulation system validation
  - TDoA algorithm testing
  - Multi-buoy correlation analysis
  - Accuracy and limitation assessment

### **System Status**
- **`test_final_system_status.py`** - Current system status reporting
  - Live buoy status
  - Signal detection status
  - Operational readiness assessment

## 🚀 **Quick Start**

### **Validate Complete System:**
```bash
cd tests
python3 test_comprehensive_system_validation.py
```

### **Run All Tests:**
```bash
python3 run_tests.py
```

### **Check Hardware:**
```bash
python3 test_hardware_detection.py
```

## 📊 **Expected Results**

- **System Validation**: 4/5 tests should pass (triangulation needs 3+ buoys)
- **Hardware Detection**: Should detect connected SDR and GPS devices
- **Integration Tests**: All communication and signal flow should work
- **TDoA Analysis**: Should identify timing and multi-buoy requirements

## 🎯 **Test Priorities**

1. **First Run**: `test_comprehensive_system_validation.py` (validates everything)
2. **Hardware Issues**: `test_hardware_detection.py` (checks connected devices)  
3. **Signal Problems**: `test_system_integration.py` (validates signal flow)
4. **Triangulation**: `test_tdoa_validation.py` (when adding more buoys) 