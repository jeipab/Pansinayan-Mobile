"""
Test script for Pansinayan server.
Tests health check and inference endpoints.
"""

import requests
import numpy as np
import time
import json

# Server URL
SERVER_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("\n" + "="*50)
    print("Testing Health Endpoint")
    print("="*50)
    
    response = requests.get(f"{SERVER_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✓ Health check passed")


def test_inference(model_type="transformer", sequence_length=150):
    """Test inference endpoint"""
    print("\n" + "="*50)
    print(f"Testing Inference Endpoint ({model_type}, T={sequence_length})")
    print("="*50)
    
    # Generate dummy keypoints
    keypoints = np.random.rand(sequence_length, 178).tolist()
    
    # Prepare request
    payload = {
        "keypoints": keypoints,
        "model_type": model_type
    }
    
    # Send request
    start_time = time.time()
    response = requests.post(f"{SERVER_URL}/predict", json=payload)
    total_time = (time.time() - start_time) * 1000
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Sequence Length: {data['sequence_length']}")
        print(f"Model Used: {data['model_used']}")
        print(f"Inference Time: {data['inference_time_ms']:.2f}ms")
        print(f"Total Time (with network): {total_time:.2f}ms")
        print(f"CTC Output Shape: {len(data['ctc_log_probs'])} x {len(data['ctc_log_probs'][0])}")
        
        if data['cat_logits']:
            print(f"Category Output Shape: {len(data['cat_logits'])} x {len(data['cat_logits'][0])}")
        
        print("✓ Inference test passed")
    else:
        print(f"Error: {response.json()}")
        raise AssertionError("Inference failed")


def test_invalid_input():
    """Test error handling with invalid input"""
    print("\n" + "="*50)
    print("Testing Error Handling")
    print("="*50)
    
    # Test 1: Wrong feature dimension
    print("\n1. Testing wrong feature dimension...")
    payload = {
        "keypoints": [[0.5] * 100] * 10,  # Wrong: 100 instead of 178
        "model_type": "transformer"
    }
    response = requests.post(f"{SERVER_URL}/predict", json=payload)
    print(f"Status: {response.status_code} (expected 422)")
    assert response.status_code == 422
    print("✓ Correctly rejected wrong dimension")
    
    # Test 2: Invalid model type
    print("\n2. Testing invalid model type...")
    payload = {
        "keypoints": [[0.5] * 178] * 10,
        "model_type": "invalid_model"
    }
    response = requests.post(f"{SERVER_URL}/predict", json=payload)
    print(f"Status: {response.status_code} (expected 422)")
    assert response.status_code == 422
    print("✓ Correctly rejected invalid model")
    
    # Test 3: Empty keypoints
    print("\n3. Testing empty keypoints...")
    payload = {
        "keypoints": [],
        "model_type": "transformer"
    }
    response = requests.post(f"{SERVER_URL}/predict", json=payload)
    print(f"Status: {response.status_code} (expected 422)")
    assert response.status_code == 422
    print("✓ Correctly rejected empty input")


def benchmark(model_type="transformer", iterations=10):
    """Benchmark inference performance"""
    print("\n" + "="*50)
    print(f"Benchmarking {model_type.upper()} Model")
    print("="*50)
    
    keypoints = np.random.rand(150, 178).tolist()
    payload = {"keypoints": keypoints, "model_type": model_type}
    
    times = []
    for i in range(iterations):
        start = time.time()
        response = requests.post(f"{SERVER_URL}/predict", json=payload)
        total_time = (time.time() - start) * 1000
        
        if response.status_code == 200:
            inference_time = response.json()["inference_time_ms"]
            times.append(inference_time)
            print(f"Iteration {i+1}: {inference_time:.2f}ms (total: {total_time:.2f}ms)")
    
    if times:
        print(f"\nResults ({iterations} iterations):")
        print(f"  Mean: {np.mean(times):.2f}ms")
        print(f"  Std:  {np.std(times):.2f}ms")
        print(f"  Min:  {np.min(times):.2f}ms")
        print(f"  Max:  {np.max(times):.2f}ms")


if __name__ == "__main__":
    try:
        print("\n" + "="*50)
        print("Pansinayan Server Test Suite")
        print("="*50)
        
        # Run tests
        test_health()
        test_inference("transformer", 150)
        test_inference("gru", 150)
        test_invalid_input()
        
        # Benchmark
        benchmark("transformer", 10)
        benchmark("gru", 10)
        
        print("\n" + "="*50)
        print("All Tests Passed! ✓")
        print("="*50)
        
    except Exception as e:
        print(f"\n✗ Test Failed: {e}")
        import traceback
        traceback.print_exc()