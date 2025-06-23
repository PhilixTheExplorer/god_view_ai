"""
Test script for the GodView Alert API
Run this after starting the API server to test alert functionality
"""

import requests
import json
from datetime import datetime

# API base URL (adjust if running on different port)
API_BASE = "http://localhost:8000"

def test_api_health():
    """Test API health endpoint"""
    try:
        response = requests.get(f"{API_BASE}/health")
        print(f"Health Check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_create_alert():
    """Test creating a new alert"""
    alert_data = {
        "patient_id": 123,
        "room_id": "ROOM_001",
        "alert_type": "FALL_DETECTED",
        "description": "Patient fall detected during test - This alert should be sent to all doctors via Telegram",
        "confidence": 0.95,
        "frame_number": 1500
    }
    
    try:
        response = requests.post(f"{API_BASE}/alerts", json=alert_data)
        print(f"Create Alert: {response.status_code}")
        result = response.json()
        print(f"Response: {result}")
        print("💬 Note: This alert should have been sent to all users with 'doctor' role in Telegram")
        return response.status_code == 200
    except Exception as e:
        print(f"Create alert failed: {e}")
        return False

def test_get_alerts():
    """Test getting all alerts"""
    try:
        response = requests.get(f"{API_BASE}/alerts")
        print(f"Get Alerts: {response.status_code}")
        alerts = response.json()
        print(f"Total alerts: {len(alerts)}")
        if alerts:
            print(f"Latest alert: {alerts[-1]}")
        return response.status_code == 200
    except Exception as e:
        print(f"Get alerts failed: {e}")
        return False

def test_get_alert_stats():
    """Test getting alert statistics"""
    try:
        response = requests.get(f"{API_BASE}/alerts/stats")
        print(f"Get Stats: {response.status_code}")
        print(f"Stats: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Get stats failed: {e}")
        return False

def test_create_test_alert():
    """Test creating a test alert"""
    try:
        response = requests.post(f"{API_BASE}/alerts/test")
        print(f"Create Test Alert: {response.status_code}")
        result = response.json()
        print(f"Response: {result}")
        print("💬 Note: This test alert should have been sent to all users with 'doctor' role in Telegram")
        return response.status_code == 200
    except Exception as e:
        print(f"Create test alert failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("GodView Alert API Test Suite")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_api_health),
        ("Create Alert", test_create_alert),
        ("Get Alerts", test_get_alerts),
        ("Get Alert Stats", test_get_alert_stats),
        ("Create Test Alert", test_create_test_alert),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        success = test_func()
        results.append((test_name, success))
        print(f"Result: {'✅ PASS' if success else '❌ FAIL'}")
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")

if __name__ == "__main__":
    main()
