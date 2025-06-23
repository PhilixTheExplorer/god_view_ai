"""
Tests for the GodView Alert API endpoints
"""

import pytest
import requests
import json
from datetime import datetime

class TestAlertAPI:
    """Test suite for Alert API endpoints"""
    
    def test_api_health(self, api_base_url):
        """Test API health endpoint"""
        try:
            response = requests.get(f"{api_base_url}/health")
            assert response.status_code == 200
            
            data = response.json()
            assert "status" in data
            assert data["status"] == "healthy"
            
            print(f"✅ Health Check: {response.status_code}")
            print(f"Response: {data}")
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running - start with 'python start_api.py'")
        except Exception as e:
            pytest.fail(f"Health check failed: {e}")
    
    def test_create_alert(self, api_base_url, test_alert_data):
        """Test creating a new alert"""
        try:
            response = requests.post(f"{api_base_url}/alerts", json=test_alert_data)
            assert response.status_code == 200
            
            result = response.json()
            assert "id" in result
            assert result["patient_id"] == test_alert_data["patient_id"]
            assert result["alert_type"] == test_alert_data["alert_type"]
            
            print(f"✅ Create Alert: {response.status_code}")
            print(f"Alert ID: {result['id']}")
            print("💬 Note: This alert should have been sent to all doctors via Telegram")
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running - start with 'python start_api.py'")
        except Exception as e:
            pytest.fail(f"Create alert failed: {e}")
    
    def test_get_alerts(self, api_base_url):
        """Test getting all alerts"""
        try:
            response = requests.get(f"{api_base_url}/alerts")
            assert response.status_code == 200
            
            alerts = response.json()
            assert isinstance(alerts, list)
            
            print(f"✅ Get Alerts: {response.status_code}")
            print(f"Total alerts: {len(alerts)}")
            
            if alerts:
                latest_alert = alerts[-1]
                assert "id" in latest_alert
                assert "alert_type" in latest_alert
                print(f"Latest alert: {latest_alert['alert_type']} (ID: {latest_alert['id']})")
                
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running - start with 'python start_api.py'")
        except Exception as e:
            pytest.fail(f"Get alerts failed: {e}")
    
    def test_get_alert_stats(self, api_base_url):
        """Test getting alert statistics"""
        try:
            response = requests.get(f"{api_base_url}/alerts/stats")
            assert response.status_code == 200
            
            stats = response.json()
            assert "total_alerts" in stats
            assert "alert_types" in stats
            assert isinstance(stats["total_alerts"], int)
            assert isinstance(stats["alert_types"], dict)
            
            print(f"✅ Get Stats: {response.status_code}")
            print(f"Total alerts: {stats['total_alerts']}")
            print(f"Alert types: {stats['alert_types']}")
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running - start with 'python start_api.py'")
        except Exception as e:
            pytest.fail(f"Get stats failed: {e}")
    
    def test_create_test_alert(self, api_base_url):
        """Test creating a test alert"""
        try:
            response = requests.post(f"{api_base_url}/alerts/test")
            assert response.status_code == 200
            
            result = response.json()
            assert "id" in result
            assert result["alert_type"] == "TEST_ALERT"
            
            print(f"✅ Create Test Alert: {response.status_code}")
            print(f"Test Alert ID: {result['id']}")
            print("💬 Note: This test alert should have been sent to all doctors via Telegram")
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running - start with 'python start_api.py'")
        except Exception as e:
            pytest.fail(f"Create test alert failed: {e}")

    def test_invalid_alert_data(self, api_base_url):
        """Test creating alert with invalid data"""
        invalid_data = {
            "patient_id": "invalid",  # Should be int
            # Missing required fields
        }
        
        try:
            response = requests.post(f"{api_base_url}/alerts", json=invalid_data)
            assert response.status_code == 422  # Validation error
            
            print(f"✅ Invalid data validation: {response.status_code}")
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running - start with 'python start_api.py'")
        except Exception as e:
            pytest.fail(f"Invalid data test failed: {e}")

def run_manual_tests():
    """
    Manual test runner for when pytest is not available
    """
    print("=" * 60)
    print("GodView Alert API Test Suite")
    print("=" * 60)
    
    api_base = "http://localhost:8000"
    test_data = {
        "patient_id": 123,
        "room_id": "TEST_ROOM_001",
        "alert_type": "FALL_DETECTED",
        "description": "Test fall detection alert",
        "confidence": 0.95,
        "frame_number": 1500
    }
    
    # Create test instance
    test_instance = TestAlertAPI()
    
    tests = [
        ("Health Check", lambda: test_instance.test_api_health(api_base)),
        ("Create Alert", lambda: test_instance.test_create_alert(api_base, test_data)),
        ("Get Alerts", lambda: test_instance.test_get_alerts(api_base)),
        ("Get Alert Stats", lambda: test_instance.test_get_alert_stats(api_base)),
        ("Create Test Alert", lambda: test_instance.test_create_test_alert(api_base)),
        ("Invalid Data", lambda: test_instance.test_invalid_alert_data(api_base)),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            test_func()
            results.append((test_name, True))
            print(f"Result: ✅ PASS")
        except Exception as e:
            results.append((test_name, False))
            print(f"Result: ❌ FAIL - {e}")
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")

if __name__ == "__main__":
    run_manual_tests()
