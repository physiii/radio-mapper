#!/usr/bin/env python3
"""
Test runner for Radio-Mapper system tests
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from test_system_integration import RadioMapperSystemTest

def main():
    print("Radio-Mapper System Test Runner")
    print("=" * 40)
    
    tester = RadioMapperSystemTest()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! System is working correctly.")
        return 0
    else:
        print("\n💥 Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 