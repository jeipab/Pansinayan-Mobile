"""
Test script for Pansinayan server.
Tests health check, inference endpoints, and error handling.
"""

import requests
import numpy as np
import time
import json
from typing import Dict, Any

# Server URL
SERVER_URL = "http://localhost:8000"


def test_health() -> None:
    """
    Test health endpoint.
    
    Validates:
    - Server responds with 200
    - Status is "healthy"
    - Models are loaded
    - Device information is present
    """
    print("\n" + "="*50)
    print("Testing Health Endpoint")
    print("="*50)
    
    response = requests.get(f"{SERVER_URL}/health", timeout=10)
    print(f"Status Code: {response.status_code}")
    
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert data["status"] == "healthy", f"Expected 'healthy', got '{data['status']}'"
    assert len(data["models_loaded"]) > 0, "No models loaded"
    assert "device" in data, "Device information missing"
    
    print("✓ Health check passed")


def test_inference(model_type: str = "transformer", sequence_length: int = 150) -> None:
    """
    Test inference endpoint with valid input.
    
    Args:
        model_type: Model to test ("transformer" or "gru")
        sequence_length: Length of keypoint sequence
    """
    print("\n" + "="*50)
    print(f"Testing Inference Endpoint ({model_type}, T={sequence_length})")
    print("="*50)
    
    # Generate dummy keypoints (normalized to [0, 1])
    keypoints = np.random.rand(sequence_length, 178).tolist()
    
    # Prepare request
    payload: Dict[str, Any] = {
        "keypoints": keypoints,
        "model_type": model_type
    }
    
    # Send request
    start_time = time.time()
    response = requests.post(f"{SERVER_URL}/predict", json=payload, timeout=60)
    total_time = (time.time() - start_time) * 1000
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Sequence Length: {data['sequence_length']}")
        print(f"Model Used: {data['model_used']}")
        print(f"Inference Time: {data['inference_time_ms']:.2f}ms")
        print(f"Total Time (with network): {total_time:.2f}ms")
        print(f"CTC Output Shape: {len(data['ctc_log_probs'])} x {len(data['ctc_log_probs'][0])}")
        
        # Validate response structure
        assert data['sequence_length'] == sequence_length, "Sequence length mismatch"
        assert data['model_used'] == model_type, "Model type mismatch"
        assert len(data['ctc_log_probs']) == sequence_length, "CTC output length mismatch"
        assert len(data['ctc_log_probs'][0]) > 0, "CTC output dimension invalid"
        
        if data.get('cat_logits'):
            print(f"Category Output Shape: {len(data['cat_logits'])} x {len(data['cat_logits'][0])}")
            assert len(data['cat_logits']) == sequence_length, "Category output length mismatch"
        
        print("✓ Inference test passed")
    else:
        error_data = response.json()
        print(f"Error: {error_data}")
        raise AssertionError(f"Inference failed: {error_data}")


def test_invalid_input() -> None:
    """
    Test error handling with various invalid inputs.
    
    Tests:
    - Wrong feature dimension
    - Inconsistent frame dimensions
    - Invalid model type
    - Empty keypoints
    - Sequence too long
    - Sequence too short (boundary)
    """
    print("\n" + "="*50)
    print("Testing Error Handling")
    print("="*50)
    
    # Test 1: Wrong feature dimension
    print("\n1. Testing wrong feature dimension...")
    payload = {
        "keypoints": [[0.5] * 100] * 10,  # Wrong: 100 instead of 178
        "model_type": "transformer"
    }
    response = requests.post(f"{SERVER_URL}/predict", json=payload, timeout=10)
    print(f"Status: {response.status_code} (expected 422)")
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    print("✓ Correctly rejected wrong dimension")
    
    # Test 2: Inconsistent frame dimensions
    print("\n2. Testing inconsistent frame dimensions...")
    keypoints = [[0.5] * 178] * 9
    keypoints.append([0.5] * 100)  # Last frame has wrong dimension
    payload = {
        "keypoints": keypoints,
        "model_type": "transformer"
    }
    response = requests.post(f"{SERVER_URL}/predict", json=payload, timeout=10)
    print(f"Status: {response.status_code} (expected 422)")
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    print("✓ Correctly rejected inconsistent dimensions")
    
    # Test 3: Invalid model type
    print("\n3. Testing invalid model type...")
    payload = {
        "keypoints": [[0.5] * 178] * 10,
        "model_type": "invalid_model"
    }
    response = requests.post(f"{SERVER_URL}/predict", json=payload, timeout=10)
    print(f"Status: {response.status_code} (expected 422)")
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    print("✓ Correctly rejected invalid model")
    
    # Test 4: Empty keypoints
    print("\n4. Testing empty keypoints...")
    payload = {
        "keypoints": [],
        "model_type": "transformer"
    }
    response = requests.post(f"{SERVER_URL}/predict", json=payload, timeout=10)
    print(f"Status: {response.status_code} (expected 422)")
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    print("✓ Correctly rejected empty input")
    
    # Test 5: Sequence too long (boundary test)
    print("\n5. Testing sequence too long (boundary)...")
    payload = {
        "keypoints": [[0.5] * 178] * 301,  # Exceeds max of 300
        "model_type": "transformer"
    }
    response = requests.post(f"{SERVER_URL}/predict", json=payload, timeout=10)
    print(f"Status: {response.status_code} (expected 400 or 422)")
    assert response.status_code in [400, 422], f"Expected 400/422, got {response.status_code}"
    print("✓ Correctly rejected sequence too long")
    
    # Test 6: Minimum valid sequence (boundary test)
    print("\n6. Testing minimum valid sequence (boundary)...")
    payload = {
        "keypoints": [[0.5] * 178] * 1,  # Minimum: 1 frame
        "model_type": "transformer"
    }
    response = requests.post(f"{SERVER_URL}/predict", json=payload, timeout=60)
    print(f"Status: {response.status_code} (expected 200)")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    print("✓ Correctly accepted minimum sequence")


def test_stats_endpoint() -> None:
    """Test stats endpoint for metrics"""
    print("\n" + "="*50)
    print("Testing Stats Endpoint")
    print("="*50)
    
    response = requests.get(f"{SERVER_URL}/stats", timeout=10)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    
    assert "models_loaded" in data, "models_loaded missing"
    assert "device" in data, "device missing"
    assert "request_metrics" in data, "request_metrics missing"
    
    print("✓ Stats endpoint test passed")


def benchmark(model_type: str = "transformer", iterations: int = 10) -> None:
    """
    Benchmark inference performance.
    
    Args:
        model_type: Model to benchmark
        iterations: Number of iterations to run
    """
    print("\n" + "="*50)
    print(f"Benchmarking {model_type.upper()} Model")
    print("="*50)
    
    keypoints = np.random.rand(150, 178).tolist()
    payload: Dict[str, Any] = {"keypoints": keypoints, "model_type": model_type}
    
    inference_times = []
    total_times = []
    
    for i in range(iterations):
        start = time.time()
        response = requests.post(f"{SERVER_URL}/predict", json=payload, timeout=60)
        total_time = (time.time() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            inference_time = data["inference_time_ms"]
            inference_times.append(inference_time)
            total_times.append(total_time)
            print(f"Iteration {i+1}: {inference_time:.2f}ms inference, {total_time:.2f}ms total")
        else:
            print(f"Iteration {i+1}: Failed with status {response.status_code}")
    
    if inference_times:
        print(f"\nInference Time Results ({iterations} iterations):")
        print(f"  Mean: {np.mean(inference_times):.2f}ms")
        print(f"  Std:  {np.std(inference_times):.2f}ms")
        print(f"  Min:  {np.min(inference_times):.2f}ms")
        print(f"  Max:  {np.max(inference_times):.2f}ms")
        
        print(f"\nTotal Time Results (including network):")
        print(f"  Mean: {np.mean(total_times):.2f}ms")
        print(f"  Min:  {np.min(total_times):.2f}ms")
        print(f"  Max:  {np.max(total_times):.2f}ms")


if __name__ == "__main__":
    """
    Run complete test suite for Pansinayan server.
    
    Tests include:
    - Health check
    - Inference with both models
    - Error handling
    - Stats endpoint
    - Performance benchmarking
    """
    try:
        print("\n" + "="*50)
        print("Pansinayan Server Test Suite")
        print("="*50)
        
        # Basic functionality tests
        test_health()
        test_stats_endpoint()
        
        # Inference tests
        test_inference("transformer", 150)
        test_inference("gru", 150)
        
        # Edge case tests
        test_invalid_input()
        
        # Performance benchmarks
        print("\n" + "="*50)
        print("Performance Benchmarks")
        print("="*50)
        benchmark("transformer", 10)
        benchmark("gru", 10)
        
        print("\n" + "="*50)
        print("All Tests Passed! ✓")
        print("="*50)
        
    except requests.exceptions.ConnectionError:
        print("\n✗ Connection Error: Is the server running?")
        print(f"   Trying to connect to: {SERVER_URL}")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)